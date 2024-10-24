import math

import torch
from torch.utils.data import DataLoader, TensorDataset


def update_models_end_to_end_unconstrained(
    train_x, train_y_scores, censoring, objective, model, mll, learning_rte, num_update_epochs
):
    """Finetune VAE end to end with surrogate model
    This method is build to be compatible with the
    SELFIES VAE interface
    """
    objective.vae.train()
    model.train()
    optimizer = torch.optim.Adam(
        [{"params": objective.vae.parameters()}, {"params": model.parameters(), "lr": learning_rte}], lr=learning_rte
    )
    # max batch size smaller to avoid memory limit with longer strings (more tokens)
    max_string_length = len(max(train_x, key=len))
    bsz = max(1, int(2560 / max_string_length))
    num_batches = math.ceil(len(train_x) / bsz)
    for _ in range(num_update_epochs):
        for batch_ix in range(num_batches):
            start_idx, stop_idx = batch_ix * bsz, (batch_ix + 1) * bsz
            batch_list = train_x[start_idx:stop_idx]
            z, vae_loss = objective.vae_forward(batch_list)
            batch_y = train_y_scores[start_idx:stop_idx]
            batch_y = torch.tensor(batch_y).float()
            pred = model(z)
            if censoring is not None:  # If censored observations
                censoring_batch = censoring[start_idx:stop_idx]
                censoring_batch = torch.tensor(censoring_batch)
                surr_loss = -mll(pred, batch_y.cuda(), censoring=censoring_batch.cuda())
            else:
                surr_loss = -mll(pred, batch_y.cuda())
            # add losses and back prop
            loss = vae_loss + surr_loss
            if loss.isfinite().item():
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(objective.vae.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                pass

    objective.vae.eval()
    model.eval()

    return objective, model


def update_models_end_to_end_with_constraints(
    train_x,
    train_y_scores,
    censoring,
    objective,
    model,
    mll,
    learning_rte,
    num_update_epochs,
    train_c_scores=None,
    c_models=[],
    c_mlls=[],
):
    """Finetune VAE end to end with surrogate model
    This method is build to be compatible with the
    SELFIES VAE interface
    """
    objective.vae.train()
    model.train()

    optimize_list = [{"params": objective.vae.parameters()}, {"params": model.parameters(), "lr": learning_rte}]
    if train_c_scores is not None:
        for c_model in c_models:
            c_model.train()
            optimize_list.append({"params": c_model.parameters(), "lr": learning_rte})
    optimizer = torch.optim.Adam(optimize_list, lr=learning_rte)

    # max batch size smaller to avoid memory limit with longer strings (more tokens)
    max_string_length = len(max(train_x, key=len))
    bsz = max(1, int(2560 / max_string_length))
    num_batches = math.ceil(len(train_x) / bsz)
    for _ in range(num_update_epochs):
        for batch_ix in range(num_batches):
            start_idx, stop_idx = batch_ix * bsz, (batch_ix + 1) * bsz
            batch_list = train_x[start_idx:stop_idx]
            z, vae_loss = objective.vae_forward(batch_list)
            batch_y = train_y_scores[start_idx:stop_idx]
            batch_y = torch.tensor(batch_y).float()
            pred = model(z)

            if censoring is not None:  # If censored observations
                censoring_batch = censoring[start_idx:stop_idx]
                censoring_batch = torch.tensor(censoring_batch)
                surr_loss = -mll(pred, batch_y.cuda(), censoring=censoring_batch.cuda())
            else:
                surr_loss = -mll(pred, batch_y.cuda())

            # add loss terms from constraint models!
            if train_c_scores is not None:
                batch_c = train_c_scores[start_idx:stop_idx]
                for ix, c_model in enumerate(c_models):
                    batch_c_ix = batch_c[:, ix]
                    c_pred_ix = c_model(z)
                    if censoring is not None:  # If censored observations
                        loss_cmodel_ix = -c_mlls[ix](c_pred_ix, batch_c_ix.cuda(), censoring=censoring_batch.cuda())
                    else:
                        loss_cmodel_ix = -c_mlls[ix](c_pred_ix, batch_c_ix.cuda())
                    surr_loss = surr_loss + loss_cmodel_ix
            # add losses and back prop
            loss = vae_loss + surr_loss
            if loss.isfinite().item():
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(objective.vae.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                pass

    objective.vae.eval()
    model.eval()
    if train_c_scores is not None:
        for c_model in c_models:
            c_model.eval()

    return objective, model


def update_surr_model(
    model,
    mll,
    learning_rte,
    train_z,
    train_y,
    censoring,
    n_epochs,
):
    model.train()
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": learning_rte}], lr=learning_rte)
    train_bsz = min(len(train_y), 128)
    if censoring is not None:
        train_dataset = TensorDataset(train_z, train_y, censoring)
    else:
        train_dataset = TensorDataset(train_z, train_y)
    train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
    for e in range(n_epochs):
        batch_n = 0
        for data in train_loader:
            batch_n += 1
            inputs = data[0]
            scores = data[1]
            if len(inputs) == 1:
                # NOTE: hack to handle case with only one training data point
                # concatenate to repeat the same point twice
                # needed bc single piont causes bug in mll w/ censored GP
                # todo later: fix this bug so hack isn't needed
                inputs = torch.cat((inputs, inputs))
                scores = torch.cat((scores, scores))
            output = model(inputs.cuda())
            if censoring is not None:  # If censored observations
                censoring_batch = data[2]
                if len(censoring_batch) == 1:
                    censoring_batch = torch.cat((censoring_batch, censoring_batch))
                loss = -mll(output, scores.cuda(), censoring=censoring_batch.cuda())
            else:
                loss = -mll(output, scores.cuda())
            if loss.isfinite().item():
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                pass
    model = model.eval()

    return model


def update_constraint_surr_models(
    c_models,
    c_mlls,
    learning_rte,
    train_z,
    train_c,
    censoring,
    n_epochs,
):
    updated_c_models = []
    for ix, c_model in enumerate(c_models):
        updated_model = update_surr_model(
            c_model, c_mlls[ix], learning_rte, train_z, train_c[:, ix], censoring, n_epochs
        )
        updated_c_models.append(updated_model)

    return updated_c_models
