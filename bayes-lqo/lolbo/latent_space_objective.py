import sys
import time

import numpy as np
import torch

sys.path.append("../")

from oracle.oracle import WorkloadSpec, _resolve_codec

torch.set_float32_matmul_precision("highest")


def canonicalize(x: list, workload_spec: WorkloadSpec):
    codec = _resolve_codec(workload_spec)
    return codec.encode(codec.decode(workload_spec.query_tables, x))


class LatentSpaceObjective:
    """Base class for any latent space optimization task
    class supports any optimization task with accompanying VAE
    such that during optimization, latent space points (z)
    must be passed through the VAE decoder to obtain
    original input space points (x) which can then
    be passed into the oracle to obtain objective values (y)"""

    def __init__(
        self,
        xs_to_scores_dict={},
        xs_to_censoring_dict={},
        num_calls=0,
        task_id="",
        init_vae=True,
    ):
        # dict used to track xs and scores (ys) queried during optimization
        self.xs_to_scores_dict = xs_to_scores_dict
        self.xs_to_censoring_dict = xs_to_censoring_dict

        # track total number of times the oracle has been called
        self.num_calls = num_calls

        # string id for optimization task, often used by oracle
        #   to differentiate between similar tasks (ie for guacamol)
        self.task_id = task_id

        # load in pretrained VAE, store in variable self.vae
        self.vae = None
        if init_vae:
            self.initialize_vae()
            assert self.vae is not None

    def __call__(self, z, timeouts_list=None, decoded_xs=None):
        """Input
            z: a numpy array or pytorch tensor of latent space points
        Output
            out_dict['valid_zs'] = the zs which decoded to valid xs
            out_dict['decoded_xs'] = an array of valid xs obtained from input zs
            out_dict['scores']: an array of valid scores obtained from input zs
            out_dict['constr_vals']: an array of constraint values or none if unconstrained
        """
        start_time_decode = time.time()
        if type(z) is np.ndarray:
            z = torch.from_numpy(z).float()

        # To support rerunning thompson sampling instead of only rejection sampling from the vae
        # we move decoding to generate_candidates if `thompson_rejection` is set to True
        if decoded_xs is None:
            decoded_xs = self.vae_decode(z)

        decoded_xs = [canonicalize(x, self.objective_function.full_workload_spec) for x in decoded_xs]

        scores = []
        cens = []
        xs_to_be_queired = []
        timeouts_to_be_queried = []
        for index, x in enumerate(decoded_xs):
            # get rid of X's (deletion)
            # if we have already computed the score, don't
            #   re-compute (don't call oracle unnecessarily)
            list_to_tuple = False
            if isinstance(x, list):
                x = tuple(x)
                list_to_tuple = True
            if x in self.xs_to_scores_dict:
                score = self.xs_to_scores_dict[x]
                cen = self.xs_to_censoring_dict[x]
            else:  # otherwise call the oracle to get score
                score = "?"
                cen = "?"
                if list_to_tuple:
                    x = list(x)
                xs_to_be_queired.append(x)
                if timeouts_list is not None:
                    timeouts_to_be_queried.append(timeouts_list[index])
            scores.append(score)
            cens.append(cen)
        TIME_DECODE = time.time() - start_time_decode

        start_time_query_oracle = time.time()
        if timeouts_list is None:
            computed_scores, computed_censoring = self.query_oracle(xs_to_be_queired)
        else:
            computed_scores, computed_censoring = self.query_oracle(xs_to_be_queired, timeouts_to_be_queried)

        TIME_WAIT_IN_QUEUE_AND_QUERY_ORACLE = time.time() - start_time_query_oracle
        TIME_QUERY_ORACLE_non_parallel = self.objective_function.total_non_parallel_runtime

        start_time_organize_results = time.time()
        # move computed scores to scores list
        temp_scores = []
        temp_censoring = []
        ix = 0
        for iz, score in enumerate(scores):
            if score == "?":
                temp_scores.append(computed_scores[ix])
                temp_censoring.append(computed_censoring[ix])
                # add score to dict so we don't have to
                #   compute it again if we get the same input x
                x_i = xs_to_be_queired[ix]
                if isinstance(x_i, list):
                    x_i = tuple(x_i)
                self.xs_to_scores_dict[x_i] = computed_scores[ix]
                self.xs_to_censoring_dict[x_i] = computed_censoring[ix]
                ix += 1
            else:
                temp_scores.append(score)
                temp_censoring.append(cens[iz])
        scores = temp_scores
        censoring = temp_censoring

        # track number of oracle calls
        #   nan scores happen when we pass an invalid
        #   molecular string and thus avoid calling the
        #   oracle entirely
        self.num_calls += (np.logical_not(np.isnan(np.array(computed_scores)))).sum()

        scores_arr = np.array(scores)
        censoring_arr = np.array(censoring)
        # decoded_xs = np.array(decoded_xs) # doesnt work when each x is a list :(

        # get valid zs, xs, and scores
        bool_arr = np.logical_not(np.isnan(scores_arr))

        # decoded_xs = decoded_xs[bool_arr]
        # Work around for when each x is a list:
        temp = []
        for ix, bool1 in enumerate(bool_arr):
            if bool1.item():
                temp.append(decoded_xs[ix])
        decoded_xs = temp

        scores_arr = scores_arr[bool_arr]
        valid_zs = z[bool_arr]
        censoring_arr = censoring_arr[bool_arr]
        TIME_ORGANIZE_RESULTS = time.time() - start_time_organize_results

        out_dict = {}
        out_dict["scores"] = scores_arr
        out_dict["valid_zs"] = valid_zs
        out_dict["decoded_xs"] = decoded_xs
        out_dict["censoring"] = torch.from_numpy(censoring_arr)
        out_dict["constr_vals"] = self.compute_constraints(decoded_xs)

        # new version:: update to separate out timings
        out_dict["time_organize_results"] = TIME_ORGANIZE_RESULTS
        out_dict["time_vae_decode"] = TIME_DECODE
        out_dict["time_query_oracle"] = TIME_QUERY_ORACLE_non_parallel
        out_dict["time_wait_in_oracle_queue"] = TIME_WAIT_IN_QUEUE_AND_QUERY_ORACLE - TIME_QUERY_ORACLE_non_parallel

        # old version::
        # out_dict['time_call'] = TIME_ORGANIZE_RESULTS + TIME_WAIT_IN_QUEUE_AND_QUERY_ORACLE + TIME_DECODE
        # out_dict['time_call_non_parallel'] = TIME_ORGANIZE_RESULTS + TIME_QUERY_ORACLE_non_parallel + TIME_DECODE

        return out_dict

    def vae_decode(self, z):
        """Input
            z: a tensor latent space points
        Output
            a corresponding list of the decoded input space
            items output by vae decoder
        """
        raise NotImplementedError("Must implement vae_decode()")

    def query_oracle(self, x):
        """Input:
            a list of input space items x (i.e. molecule strings)
        Output:
            method queries the oracle and returns
            the corresponding list of scores y,
            or np.nan in the case that x is an invalid input,
            as well as a list of censoring indications 0/1 for each score
                1 means the observation is censroed
        """
        raise NotImplementedError("Must implement query_oracle() specific to desired optimization task")

    def initialize_vae(self):
        """Sets variable self.vae to the desired pretrained vae"""
        raise NotImplementedError("Must implement method initialize_vae() to load in vae for desired optimization task")

    def vae_forward(self, xs_batch):
        """Input:
            a list xs
        Output:
            z: tensor of resultant latent space codes
                obtained by passing the xs through the encoder
            vae_loss: the total loss of a full forward pass
                of the batch of xs through the vae
                (ie reconstruction error)
        """
        raise NotImplementedError("Must implement method vae_forward() (forward pass of vae)")

    def compute_constraints(self, xs_batch):
        """Input:
            a list xs
        Output:
            c: tensor of size (len(xs),n_constraints) of
                resultant constraint values, or
                None of problem is unconstrained
                Note: constraints, must be of form c(x) <= 0!
        """
        return None
