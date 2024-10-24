import copy
import math
import sys
import time

import gpytorch
import numpy as np
import torch
from gpytorch.mlls import PredictiveLogLikelihood

sys.path.append("../")
from lolbo.utils.bo_utils.censored_likelihood import CensoredGaussianLikelihood
from lolbo.utils.bo_utils.ppgpr import GPModelDKL
from lolbo.utils.bo_utils.turbo import TurboState, generate_batch, update_state
from lolbo.utils.utils import (
    update_constraint_surr_models,
    update_models_end_to_end_with_constraints,
    update_surr_model,
)

torch.set_float32_matmul_precision("highest")


class LOLBOState:
    def __init__(
        self,
        objective,
        train_x,
        train_y,
        train_z,
        censoring,
        censored_obs_is_max,
        train_c=None,
        k=1_000,
        num_update_epochs=2,
        init_n_epochs=20,
        learning_rte=0.01,
        bsz=1,
        best_possible_score=0.0,
        acq_func="ts",
        verbose=True,
        timeout_percentile=0.9,
        gp_stdev_multiplier=2,
        timeout_strategy="ours",
        constant_timeout=1_000_000_000,
        model_feature_extractor=None,
        shared_multiask_model=None,
        vanilla_bo=False,
        thompson_rejection=True,
    ):
        self.timeout_strategy = timeout_strategy  # strategy used to set new timeouts for db query objectives
        self.gp_stdev_multiplier = (
            gp_stdev_multiplier  # value we muptiply gp stdev by to set new timeout with gp timeout strategy
        )
        self.timeout_percentile = (
            timeout_percentile  # percentile used if we set timeout by percentile for db query objectives
        )
        self.objective = objective  # objective with vae for particular task
        self.train_x = train_x  # initial train x data
        self.train_y = train_y  # initial train y data
        self.train_z = train_z  # initial train z data
        self.train_c = train_c  # initial constraint values data
        self.k = k  # track and update on top k scoring points found
        self.num_update_epochs = num_update_epochs  # num epochs update models
        self.init_n_epochs = init_n_epochs  # num epochs train surr model on initial data
        self.learning_rte = learning_rte  # lr to use for model updates
        self.bsz = bsz  # acquisition batch size
        self.acq_func = acq_func  # acquisition function (Expected Improvement (ei) or Thompson Sampling (ts))
        self.verbose = verbose
        self.censoring = (
            censoring  # binary tensor of init censoring for censored data (1--> censored obs, 0--> uncesnored)
        )
        self.censored_obs_is_max = censored_obs_is_max  # if True, censored obs give max possible value (true value could be lower). If false, opposite
        self.model_feature_extractor = model_feature_extractor  # feature extractor for surrogate model, if none we use default linear layers for DKL
        self.vanilla_bo = vanilla_bo  # if true, use vanilla (non-turbo) optimization
        self.shared_multiask_model = (
            shared_multiask_model  # multi-task model shared with other lolbo objectives for multi-task bo
        )

        self.thompson_rejection = (
            thompson_rejection  # cross join rejection sampling for thompson sampling instead of just VAE decode
        )
        self.thompson_decoded_xs = None

        if thompson_rejection and bsz != 1:
            raise ValueError("Thompson rejection sampling only works with batch size of 1")

        if shared_multiask_model is not None:
            assert 0, "edits must be made for supporting shared multitask model (must pass in task ids everywhere and in ts, etc...)"

        self.timeouts_for_log = []
        self.absolute_min = self.train_y.min().item()
        self.absolute_range = best_possible_score - self.absolute_min

        if timeout_strategy == "constant":
            self.objective.objective_function.timeout = constant_timeout

        assert acq_func in ["ei", "ts"]

        self.progress_fails_since_last_e2e = 0
        self.tot_num_e2e_updates = 0
        self.num_re_inits = 0
        # self.best_score_seen = torch.max(train_y)
        # self.best_x_seen = train_x[torch.argmax(train_y.squeeze())]
        self.initial_model_training_complete = (
            False  # initial training of surrogate model uses all data for more epochs
        )
        self.new_best_found = False

        self.initialize_top_k()
        self.initialize_surrogate_model()
        self.initialize_tr_state()
        self.initialize_xs_to_scores_dict()

    def initialize_xs_to_scores_dict(
        self,
    ):
        # put initial xs and ys in dict to be tracked by objective
        init_xs_to_scores_dict = {}
        init_xs_to_censoring_dict = {}
        for idx, x in enumerate(self.train_x):
            if isinstance(x, list):
                x = tuple(x)
            init_xs_to_scores_dict[x] = self.train_y.squeeze()[idx].item()
            if self.censoring is not None:
                init_xs_to_censoring_dict[x] = self.censoring.squeeze()[idx].item()
        self.objective.xs_to_scores_dict = init_xs_to_scores_dict
        self.objective.xs_to_censoring_dict = init_xs_to_censoring_dict

    def initialize_top_k(self):
        """Initialize top k x, y, and zs"""
        # if we have constriants, the top k are those that meet constraints!
        if self.train_c is not None:
            bool_arr = torch.all(self.train_c <= 0, dim=-1)  # all constraint values <= 0
            vaid_train_y = self.train_y[bool_arr]
            valid_train_z = self.train_z[bool_arr]

            # valid_train_x = np.array(self.train_x)[bool_arr] # doesn't work when each x is a list
            valid_train_x = []
            for ix, bool1 in enumerate(bool_arr):
                if bool1.item():
                    temp.append(self.train_x[ix])

            valid_train_c = self.train_c[bool_arr]
            if self.censoring is not None:
                valid_censoring = self.censoring[bool_arr]
        else:
            vaid_train_y = self.train_y
            valid_train_z = self.train_z
            valid_train_x = self.train_x
            if self.censoring is not None:
                valid_censoring = self.censoring

        # If we have censoring, init top k are those that are uncensored
        if self.censoring is not None:
            bool_arr = (valid_censoring == 0).squeeze()
            vaid_train_y = vaid_train_y[bool_arr]
            valid_train_z = valid_train_z[bool_arr]
            temp = []
            for ix, bool1 in enumerate(bool_arr):
                if bool1.item():
                    temp.append(valid_train_x[ix])
            valid_train_x = temp
            if self.train_c is not None:
                valid_train_c = valid_train_c[bool_arr]
            valid_censoring = valid_censoring[bool_arr]

        if len(vaid_train_y) > 1:
            self.best_score_seen = torch.max(vaid_train_y)
            self.best_x_seen = valid_train_x[torch.argmax(vaid_train_y.squeeze())]

            # track top k scores found
            self.top_k_scores, top_k_idxs = torch.topk(vaid_train_y.squeeze(), min(self.k, vaid_train_y.shape[0]))
            self.top_k_scores = self.top_k_scores.tolist()
            top_k_idxs = top_k_idxs.tolist()
            self.top_k_xs = [valid_train_x[i] for i in top_k_idxs]
            self.top_k_zs = [valid_train_z[i].unsqueeze(-2) for i in top_k_idxs]
            if self.train_c is not None:
                self.top_k_cs = [valid_train_c[i].unsqueeze(-2) for i in top_k_idxs]
            if self.censoring is not None:
                self.top_k_censoring = [valid_censoring[i].item() for i in top_k_idxs]
        elif len(vaid_train_y) == 1:
            self.best_score_seen = vaid_train_y.item()
            self.best_x_seen = valid_train_x.item() if isinstance(valid_train_x, torch.Tensor) else valid_train_x[0]
            self.top_k_scores = [self.best_score_seen]
            self.top_k_xs = [self.best_x_seen]
            self.top_k_zs = [valid_train_z]
            if self.train_c is not None:
                self.top_k_cs = [valid_train_c]
            if self.censoring is not None:
                self.top_k_censoring = [valid_censoring]
        else:  # Everything breaks in this case (i.e. cant set timeout, cant update_next because top_k_scores is empty, ...) so I'm taking top 1
            print("WARNING: No valid points found in initialization data")
            self.best_score_seen = torch.max(self.train_y).item()
            self.best_x_seen = self.train_x[torch.argmax(self.train_y.squeeze())]

            # track top k scores found
            self.top_k_scores, top_k_idxs = torch.topk(self.train_y.squeeze(), 1)
            self.top_k_scores = self.top_k_scores.tolist()
            top_k_idxs = top_k_idxs.tolist()
            self.top_k_xs = [self.train_x[i] for i in top_k_idxs]
            self.top_k_zs = [self.train_z[i].unsqueeze(-2) for i in top_k_idxs]

            if self.train_c is not None:
                self.top_k_cs = []
            if self.censoring is not None:
                self.top_k_censoring = [self.censoring[i].item() for i in top_k_idxs]

    def initialize_tr_state(self):
        if self.train_c is not None:  # if constrained
            bool_arr = torch.all(self.train_c <= 0, dim=-1)  # all constraint values <= 0
            vaid_train_y = self.train_y[bool_arr]
            valid_c_vals = self.train_c[bool_arr]
        else:
            vaid_train_y = self.train_y
            best_constraint_values = None

        if len(vaid_train_y) == 0:
            best_value = -torch.inf
            if self.train_c is not None:
                best_constraint_values = torch.ones(1, self.train_c.shape[1]) * torch.inf
        else:
            best_value = torch.max(vaid_train_y).item()
            if self.train_c is not None:
                best_constraint_values = valid_c_vals[torch.argmax(vaid_train_y)]
                if len(best_constraint_values.shape) == 1:
                    best_constraint_values = best_constraint_values.unsqueeze(-1)
        # initialize turbo trust region state
        self.tr_state = TurboState(  # initialize turbo state
            dim=self.train_z.shape[-1],
            batch_size=self.bsz,
            best_value=best_value,
            best_constraint_values=best_constraint_values,
        )

        return self

    def initialize_constraint_surrogates(self):
        self.c_models = []
        self.c_mlls = []
        for i in range(self.train_c.shape[1]):
            if self.censoring is None:
                likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
            else:
                likelihood = CensoredGaussianLikelihood(censored_obs_is_max=self.censored_obs_is_max).cuda()
            n_pts = min(self.train_z.shape[0], 1024)
            dim = self.train_z.shape[-1]
            c_model = GPModelDKL(
                self.train_z[:n_pts, :].cuda(),
                likelihood=likelihood,
                hidden_dims=(dim, dim),
            ).cuda()
            c_mll = PredictiveLogLikelihood(c_model.likelihood, c_model, num_data=self.train_z.size(-2))
            c_model = c_model.eval()
            # c_model = self.model.cuda()
            self.c_models.append(c_model)
            self.c_mlls.append(c_mll)
        return self

    def initialize_surrogate_model(self):
        if self.shared_multiask_model is None:
            if self.censoring is None:
                likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
            else:
                likelihood = CensoredGaussianLikelihood(censored_obs_is_max=self.censored_obs_is_max).cuda()
            n_pts = min(self.train_z.shape[0], 1024)
            dim = self.train_z.shape[-1]
            self.model = GPModelDKL(
                inducing_points=self.train_z[-n_pts:, :].cuda(),
                likelihood=likelihood,
                hidden_dims=(dim, dim),
                feature_extractor=self.model_feature_extractor,
            ).cuda()
            self.model = self.model.eval()
            self.model = self.model.cuda()
        else:
            self.model = self.shared_multiask_model
        self.mll = PredictiveLogLikelihood(self.model.likelihood, self.model, num_data=self.train_z.size(-2))

        if self.train_c is not None:
            self.initialize_constraint_surrogates()

        return self

    def update_next(self, z_next_, y_next_, x_next_, censoring_next_, c_next_=None, acquisition=False):
        """Add new points (z_next, y_next, x_next) to train data
        and update progress (top k scores found so far)
        and update trust region state
        """

        if c_next_ is not None:
            if len(c_next_.shape) == 1:
                c_next_ = c_next_.unsqueeze(-1)
            valid_points = torch.all(c_next_ <= 0, dim=-1)  # all constraint values <= 0
        else:
            valid_points = torch.tensor([True] * len(y_next_))
        z_next_ = z_next_.detach().cpu()
        y_next_ = y_next_.detach().cpu()
        if len(y_next_.shape) > 1:
            y_next_ = y_next_.squeeze()
        if self.censoring is not None:
            censoring_next_ = censoring_next_.detach().cpu()
            if len(censoring_next_.shape) > 1:
                censoring_next_ = censoring_next_.squeeze()
        if len(z_next_.shape) == 1:
            z_next_ = z_next_.unsqueeze(0)
        progress = False
        for i, score in enumerate(y_next_):
            self.train_x.append(x_next_[i])

            if self.censoring is not None:
                censored_next_i = censoring_next_[i].item()
            else:
                censored_next_i = False

            # Get Index of current worst point in top k to potentially be replaced
            #   current worst = min score from censored if any are censored, otherwise just min
            #   Add constant to uncensored data so it always has higher scores than censored
            temp = copy.deepcopy(self.top_k_scores)
            if self.censoring is not None:
                curr_max = max(self.top_k_scores)
                temp = np.array(temp)
                uncensoring_bool_arr = (
                    np.array([x if not isinstance(x, torch.Tensor) else x.item() for x in self.top_k_censoring]) == 0
                )
                temp[uncensoring_bool_arr] = temp[uncensoring_bool_arr] + curr_max
                temp = temp.tolist()
            min_score = min(temp)
            min_idx = temp.index(min_score)
            actual_worst_score = self.top_k_scores[min_idx]
            if self.censoring is not None:
                worst_is_censored = self.top_k_censoring[min_idx]
            else:
                worst_is_censored = False

            if valid_points[i]:  # if y is valid according to constraints
                if len(self.top_k_scores) < self.k:
                    # if we don't yet have k top scores, add it to the list
                    self.top_k_scores.append(score.item())
                    if self.censoring is not None:
                        self.top_k_censoring.append(censored_next_i)
                    self.top_k_xs.append(x_next_[i])
                    self.top_k_zs.append(z_next_[i].unsqueeze(-2))
                    if self.train_c is not None:  # if constrained, update best constraints too
                        self.top_k_cs.append(c_next_[i].unsqueeze(-2))
                # If new is not censored and worst is censored, replace N Y
                elif (not censored_next_i) and worst_is_censored:
                    self.top_k_scores[min_idx] = score.item()
                    self.top_k_xs[min_idx] = x_next_[i]
                    self.top_k_zs[min_idx] = z_next_[i].unsqueeze(-2)  # .cuda()
                    if self.train_c is not None:  # if constrained, update best constraints too
                        self.top_k_cs[min_idx] = c_next_[i].unsqueeze(-2)
                    if self.censoring is not None:
                        self.top_k_censoring[min_idx] = censored_next_i
                # If next and worst are the same, compare scores directly
                elif censored_next_i == worst_is_censored:
                    if (
                        score.item() > actual_worst_score
                    ):  # and (x_next_[i] not in self.top_k_xs): NOTE: removed repeats check bc can't check if list is in list of lists for databases...
                        # if the score is better than the worst score in the top k list, upate the list
                        self.top_k_scores[min_idx] = score.item()
                        self.top_k_xs[min_idx] = x_next_[i]
                        self.top_k_zs[min_idx] = z_next_[i].unsqueeze(-2)  # .cuda()
                        if self.train_c is not None:  # if constrained, update best constraints too
                            self.top_k_cs[min_idx] = c_next_[i].unsqueeze(-2)
                        if self.censoring is not None:
                            self.top_k_censoring[min_idx] = censored_next_i
                # If new is censored and worst is not censored  Y N
                elif censored_next_i and (not worst_is_censored):
                    pass  # don't update, censored always worse than uncensored
                else:
                    assert 0, "Should not be possible..."

                # Only update best with uncensored points
                if not censored_next_i:
                    # if this is the first valid example we've found, OR if we imporve
                    if (self.best_score_seen is None) or (score.item() > self.best_score_seen):
                        self.progress_fails_since_last_e2e = 0
                        progress = True
                        self.best_score_seen = score.item()  # update best
                        self.best_x_seen = x_next_[i]
                        self.new_best_found = True
        if (not progress) and acquisition:  # if no progress msde, increment progress fails
            self.progress_fails_since_last_e2e += 1
        y_next_ = y_next_.unsqueeze(-1)
        if acquisition:
            self.tr_state = update_state(
                state=self.tr_state,
                Y_next=y_next_,
                C_next=c_next_,
            )
            if self.tr_state.restart_triggered:
                self.initialize_tr_state()
        self.train_z = torch.cat((self.train_z, z_next_), dim=-2)
        self.train_y = torch.cat((self.train_y, y_next_), dim=-2)
        if c_next_ is not None:
            self.train_c = torch.cat((self.train_c, c_next_), dim=-2)
        if self.censoring is not None:
            censoring_next_ = censoring_next_.unsqueeze(-1)
            self.censoring = torch.cat((self.censoring, censoring_next_), dim=-2)

        return self

    def update_surrogate_model(self):
        if not self.initial_model_training_complete:
            # first time training surr model --> train on all data
            n_epochs = self.init_n_epochs
            train_z = self.train_z
            train_y = self.train_y.squeeze(-1)
            train_c = self.train_c
            censoring = self.censoring
        else:
            # otherwise, only train on most recent batch of data
            n_epochs = self.num_update_epochs
            train_z = self.train_z[-self.bsz :]
            train_y = self.train_y[-self.bsz :].squeeze(-1)
            if self.train_c is not None:
                train_c = self.train_c[-self.bsz :]
            else:
                train_c = None
            if self.censoring is not None:
                censoring = self.censoring[-self.bsz :]
            else:
                censoring = None
        train_y_normalized = (train_y - self.absolute_min) / self.absolute_range
        try:
            self.model = update_surr_model(
                model=self.model,
                mll=self.mll,
                learning_rte=self.learning_rte,
                train_z=train_z,
                train_y=train_y_normalized,  # train_y,
                censoring=censoring,
                n_epochs=n_epochs,
            )
            self.initial_model_training_complete = True
        except:  # noqa: E722
            # Sometimes due to unstable training/ inf loss, we get
            #   errors where model params become nan, in this case we want
            #   to re-init the model on all data
            self.num_re_inits += 1
            self.initialize_surrogate_model()
            self.initial_model_training_complete = False
            self.learning_rte = self.learning_rte / 2

        if self.train_c is not None:
            self.c_models = update_constraint_surr_models(
                c_models=self.c_models,
                c_mlls=self.c_mlls,
                learning_rte=self.learning_rte,
                train_z=train_z,
                train_c=train_c,
                censoring=censoring,
                n_epochs=n_epochs,
            )

        return self

    def update_models_e2e(self):
        """Finetune VAE end to end with surrogate model"""
        self.progress_fails_since_last_e2e = 0
        new_xs = self.train_x[-self.bsz :]
        new_ys = self.train_y[-self.bsz :].squeeze(-1).tolist()
        train_x = new_xs + self.top_k_xs
        train_y = torch.tensor(new_ys + self.top_k_scores).float()
        if self.censoring is not None:
            new_censoring = self.censoring[-self.bsz :].squeeze(-1).tolist()
            censoring = torch.tensor(new_censoring + self.top_k_censoring)
        else:
            censoring = None
        c_models = []
        c_mlls = []
        train_c = None
        if self.train_c is not None:
            c_models = self.c_models
            c_mlls = self.c_mlls
            new_cs = self.train_c[-self.bsz :]
            # Note: self.top_k_cs is a list of (1, n_cons) tensors
            if len(self.top_k_cs) > 0:
                top_k_cs_tensor = torch.cat(self.top_k_cs, -2).float()
                train_c = torch.cat((new_cs, top_k_cs_tensor), -2).float()
            else:
                train_c = new_cs
            # train_c = torch.tensor(new_cs + self.top_k_cs).float()

        train_y_normalized = (train_y - self.absolute_min) / self.absolute_range
        self.objective, self.model = update_models_end_to_end_with_constraints(
            train_x=train_x,
            train_y_scores=train_y_normalized,
            censoring=censoring,
            objective=self.objective,
            model=self.model,
            mll=self.mll,
            learning_rte=self.learning_rte,
            num_update_epochs=self.num_update_epochs,
            train_c_scores=train_c,
            c_models=c_models,
            c_mlls=c_mlls,
        )
        self.tot_num_e2e_updates += 1

        return self

    def recenter(self):
        """Pass SELFIES strings back through
        VAE to find new locations in the
        new fine-tuned latent space
        """
        self.objective.vae.eval()
        self.model.train()

        optimize_list = [{"params": self.model.parameters(), "lr": self.learning_rte}]
        if self.train_c is not None:
            for c_model in self.c_models:
                c_model.train()
                optimize_list.append({"params": c_model.parameters(), "lr": self.learning_rte})
        optimizer1 = torch.optim.Adam(optimize_list, lr=self.learning_rte)
        new_xs = self.train_x[-self.bsz :]
        train_x = new_xs + self.top_k_xs
        max_string_len = len(max(train_x, key=len))
        # max batch size smaller to avoid memory limit
        #   with longer strings (more tokens)
        bsz = max(1, int(2560 / max_string_len))
        num_batches = math.ceil(len(train_x) / bsz)
        for _ in range(self.num_update_epochs):
            for batch_ix in range(num_batches):
                optimizer1.zero_grad()
                with torch.no_grad():
                    start_idx, stop_idx = batch_ix * bsz, (batch_ix + 1) * bsz
                    batch_list = train_x[start_idx:stop_idx]
                    z, _ = self.objective.vae_forward(batch_list)
                    out_dict = self.objective(z)
                    scores_arr = out_dict["scores"]
                    constraints_tensor = out_dict["constr_vals"]
                    censoring_tensor = out_dict["censoring"]
                    valid_zs = out_dict["valid_zs"]
                    xs_list = out_dict["decoded_xs"]
                if len(scores_arr) > 0:  # if some valid scores
                    scores_arr = torch.from_numpy(scores_arr)
                    pred = self.model(valid_zs)
                    scores_normalized = (scores_arr - self.absolute_min) / self.absolute_range
                    if self.censoring is None:
                        # loss = -self.mll(pred, scores_arr.cuda())
                        loss = -self.mll(pred, scores_normalized.cuda())
                    else:
                        # loss = -self.mll(pred, scores_arr.cuda(), censoring=censoring_tensor.cuda())
                        loss = -self.mll(pred, scores_normalized.cuda(), censoring=censoring_tensor.cuda())
                    if self.train_c is not None:
                        for ix, c_model in enumerate(self.c_models):
                            pred2 = c_model(valid_zs.cuda())
                            if censoring is None:
                                loss += -self.c_mlls[ix](pred2, constraints_tensor[:, ix].cuda())
                            else:
                                loss += -self.c_mlls[ix](
                                    pred2, constraints_tensor[:, ix].cuda(), censoring=censoring_tensor.cuda()
                                )
                    optimizer1.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer1.step()
                    with torch.no_grad():
                        z = z.detach().cpu()
                        self.update_next(
                            z_next_=z,  # torch.Size([75, 32])
                            y_next_=scores_arr,  # torch.Size([75])
                            x_next_=xs_list,  # list of 75 lists
                            censoring_next_=censoring_tensor,  # torch.Size([75])
                            c_next_=constraints_tensor,  # None
                        )
            torch.cuda.empty_cache()
        self.model.eval()
        if self.train_c is not None:
            for c_model in self.c_models:
                c_model.eval()

        return self

    def set_timeout_by_percentile(
        self,
    ):
        # top percentile timeout
        assert self.timeout_strategy == "percentile"
        if self.censoring is not None:
            uncensored_ys = self.train_y[self.censoring == 0]
        else:
            uncensored_ys = self.train_y
        uncensored_ys = uncensored_ys.squeeze()
        # Remember we negate the timeouts so we can treat this as a maximization problem...
        #   so we need to multiply by -1 to get the actual runtimes...
        uncensored_ys = uncensored_ys * -1
        timeout = torch.quantile(uncensored_ys, q=self.timeout_percentile).item()
        self.objective.objective_function.timeout = timeout
        self.timeouts_for_log = [timeout]

    def generate_candidates(
        self,
    ):
        """Generate a batch of candidates in
        trust region using surrogate model
        """
        start_generate_candidates = time.time()
        if self.train_c is not None:  # if constrained
            constraint_model_list = self.c_models
        else:
            constraint_model_list = None
        train_y_normalized = (self.train_y - self.absolute_min) / self.absolute_range
        self.z_next = generate_batch(
            state=self.tr_state,
            model=self.model,
            X=self.train_z,
            Y=train_y_normalized,  # self.train_y,
            batch_size=self.bsz,
            acqf=self.acq_func,
            constraint_model_list=constraint_model_list,
            vanilla_bo=self.vanilla_bo,
        )
        self.time_generate_candidates = time.time() - start_generate_candidates

    def set_db_objective_timeout(
        self,
    ):
        start_set_oracle_timeout = time.time()
        self.timeouts_next = None
        if self.timeout_strategy == "percentile":
            # reset timeout based on some percentile of all gathered data
            self.set_timeout_by_percentile()
        elif self.timeout_strategy == "constant":
            pass  # set once to constant at beginning then we are done (no dynamic changes)
        elif self.timeout_strategy == "ours":
            self.set_timeout_by_gp_uncertainty()
        else:
            assert 0, f"timeout_strategy {self.timeout_strategy} not recognized"
        self.time_set_oracle_timeout = time.time() - start_set_oracle_timeout

    def set_timeout_by_gp_uncertainty(
        self,
    ):
        max_n_steps = 20
        n_new_data_points = self.z_next.shape[0]
        best_taus = torch.zeros(n_new_data_points)
        best_runtime = self.best_score_seen * -1
        noise = self.model.likelihood.noise
        min_buffer = 1.0  # min amount larger than best to set timeout to (we can afford an extra second at least)
        if noise < min_buffer:
            noise = noise + min_buffer
        tau_min = best_runtime + noise
        tau_max = best_runtime + noise * 10
        step_sz = (tau_max - tau_min) / max_n_steps  # w/ 20 steps, steps are size noise/2, tensor([21.8220]
        tau = tau_min
        while (best_taus != 0).sum() < n_new_data_points:  # while not all taus selected
            if tau > tau_max:
                break
            temp_model = copy.deepcopy(self.model)
            train_y = torch.tensor([-tau] * n_new_data_points).cuda()
            train_y_normalized = (train_y - self.absolute_min) / self.absolute_range
            temp_model = update_surr_model(
                model=temp_model,
                mll=self.mll,
                learning_rte=self.learning_rte,
                train_z=self.z_next.cuda(),
                train_y=train_y_normalized,
                censoring=torch.tensor([[1.0]] * len(self.z_next)).cuda(),
                n_epochs=self.num_update_epochs,
            )
            pred = temp_model(self.z_next.cuda())
            two_stddevs_better_than_mean_normed = pred.mean + self.gp_stdev_multiplier * pred.stddev
            two_stddevs_better_than_mean = two_stddevs_better_than_mean_normed * self.absolute_range + self.absolute_min
            good_timeouts = two_stddevs_better_than_mean < self.best_score_seen
            for ix, good_timeout in enumerate(good_timeouts):
                if good_timeout:
                    if best_taus[ix] == 0:  # if tau hasn't already been selected
                        best_taus[ix] = tau
            tau = tau + step_sz

        best_taus[best_taus == 0] = tau_max
        best_taus = best_taus.tolist()
        self.timeouts_next = best_taus
        self.timeouts_for_log = best_taus

    def evaluate_candidates(
        self,
    ):
        # 2. Evaluate the batch of candidates by calling oracle
        with torch.no_grad():
            self.out_dict = self.objective(
                z=self.z_next, timeouts_list=self.timeouts_next, decoded_xs=self.thompson_decoded_xs
            )

            if self.thompson_rejection:
                self.out_dict["time_vae_decode"] = self.time_vae_decode_rej

        # old version, not separated
        # self.time_call_oracle = self.out_dict["time_call"]
        # self.time_call_oracle_non_parallel = self.out_dict["time_call_non_parallel"]

        # new version, separate out vae decode and results organization
        self.time_organize_results = self.out_dict["time_organize_results"]
        self.time_vae_decode = self.out_dict["time_vae_decode"]
        self.time_query_oracle = self.out_dict["time_query_oracle"]
        self.time_wait_in_oracle_queue = self.out_dict["time_wait_in_oracle_queue"]

    def update_data_w_new_points(
        self,
    ):
        """Add new evaluated points to dataset (update_next)"""
        start_update_dataset_w_new_points = time.time()
        y_next = self.out_dict["scores"]
        if len(y_next) != 0:
            y_next = torch.from_numpy(y_next).float()
            self.update_next(
                z_next_=self.out_dict["valid_zs"],
                y_next_=y_next,
                x_next_=self.out_dict["decoded_xs"],
                censoring_next_=self.out_dict["censoring"],
                c_next_=self.out_dict["constr_vals"],
                acquisition=True,
            )
        else:
            self.progress_fails_since_last_e2e += 1
            if self.verbose:
                print("GOT NO VALID Y_NEXT TO UPDATE DATA, RERUNNING ACQUISITOIN...")
        self.time_update_dataset_w_new_points = time.time() - start_update_dataset_w_new_points

    def acquisition(self):
        """1. Generate new candidate points,
        2. set timeout for db oracle
        3. call oracle to evalute new points
        4. update data with new points
        """
        if self.thompson_rejection:
            self.generate_candidates_rejection()
        else:
            self.generate_candidates()

        self.set_db_objective_timeout()
        self.evaluate_candidates()
        self.update_data_w_new_points()

    def generate_candidates_rejection(self):
        time_gen_cand = 0
        time_vae_decode = 0

        train_y_normalized = (self.train_y - self.absolute_min) / self.absolute_range

        def select_noncross(samples):
            is_cross = [self.objective.has_cross_joins(s) for s in samples]
            if all(is_cross):
                return None
            idx = is_cross.index(False)
            return idx

        found_noncross = False
        for _ in range(50):
            _start = time.time()
            z_next = generate_batch(
                state=self.tr_state,
                model=self.model,
                X=self.train_z,
                Y=train_y_normalized,
                batch_size=20,
                acqf=self.acq_func,
                vanilla_bo=self.vanilla_bo,
            )
            time_gen_cand += time.time() - _start

            _start = time.time()
            samples = self.objective.vae.sample(z_next)
            time_vae_decode += time.time() - _start

            noncross_idx = select_noncross(samples)
            if noncross_idx is None:
                continue

            self.thompson_decoded_xs = [samples[noncross_idx]]
            self.z_next = z_next[noncross_idx].view(1, -1)
            found_noncross = True
            break

        if not found_noncross:
            # Re-encode to get a sample so we dont have identical z's
            with torch.no_grad():
                z, _ = self.objective.vae_forward(
                    [
                        self.objective.worst_init_x,
                    ]
                )
            z = z.view(1, -1)
            self.z_next = z
            self.thompson_decoded_xs = [
                self.objective.worst_init_x,
            ]

        self.time_generate_candidates = time_gen_cand
        self.time_vae_decode_rej = time_vae_decode
