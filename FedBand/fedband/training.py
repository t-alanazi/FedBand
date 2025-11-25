
import copy
import time
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import (
    DEVICE,
    NUM_CLIENTS,
    LOCAL_EPOCHS,
    NUM_ROUNDS,
    BANDWIDTH_LIMIT,
    INITIAL_ROUNDS,
    ALLOCATION_METRIC,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    DIRICHLET_ALPHA,
)

from .models import CifarClientModel, ResNet1D
from .data_cifar10 import build_clients_from_cifar10_split, get_global_test_loader, print_client_class_dist
from .data_utmobilenet import build_utmobilenet_clients, print_class_counts
from .server import ServerAggregator
from .compression import (
    compress_delta_topk_by_bytes,
    _sd_zero_like,
    VALUE_BYTES,
    INDEX_BYTES,
    MIN_NNZ,
    get_or_compute_compression,
    compute_old_bits_per_entry, 
)


def clients_sync(global_model, client_models):
    """Push global weights to all clients."""
    for client_model in client_models.values():
        client_model.load_state_dict(global_model.state_dict())


def estimate_grad_norm(model, data_loader, criterion, device, max_batches=1):
    """
    Compute L2 norm of gradients on up to `max_batches` from data_loader,
    with the model synchronized to the current global weights.
    """
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)

    n_done = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        out = model(inputs)
        loss = criterion(out, targets)
        loss.backward()
        n_done += 1
        if n_done >= max_batches:
            break

    # L2 over all parameter grads
    tot = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            tot += float(g.float().pow(2).sum().item())
    gn = math.sqrt(max(tot, 0.0))

    # clean up
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    if was_training:
        model.train()
    return gn


def test_accuracy(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def calculate_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
    return total_loss / max(1, len(data_loader))


@torch.no_grad()
def allocation_from_scores(score_dict, B_bytes: float, bytes_full: float, eps: float = 1e-12):
    """
    Map nonnegative per-client scores -> byte allocation that sums to B_bytes
    Higher score => higher bytes.
    """
    N = len(score_dict)
    cids = list(range(N))

    scores = []
    for cid in cids:
        v = float(score_dict.get(cid, 0.0))
        if not np.isfinite(v) or v < 0:
            v = 0.0
        scores.append(v)

    S = float(np.sum(scores))
    if S <= eps:
        w = np.full(N, 1.0 / N, dtype=np.float64)  # uniform fallback
    else:
        w = np.array(scores, dtype=np.float64) / S

    planned = (w * float(B_bytes)).astype(np.float64)
    fractions = np.clip(planned / float(bytes_full), 0.0, 1.0)
    fractions_dict = {cid: float(fractions[i]) for i, cid in enumerate(cids)}
    return planned.tolist(), fractions_dict, float(planned.sum())


def _mb(x):
    return float(x) / 1e6


def clients_update(
    selected_clients,
    client_models,
    client_optimizers,
    client_dataloaders,
    criterion,
    device,
    *,
    fractions=None,        # dict[cid] → α in [0,1]
    byte_budgets=None,     # dict[cid] → int bytes
    global_state=None,     # required in sparse mode
    densify_for_aggregator=True,
    dense_mode=False,
    bytes_full,
    pbar=None,
):
    """
    Client updates.

    Sparse mode:
      - Computes delta = local - global_state.
      - Compresses delta with top-k by bytes.
      - Densifies again for aggregator (global + û).
    """
    client_updates = {}
    train_losses = {}
    used_bytes_per_client = {}
    train_stats = {"steps_per_client": {}, "examples_per_client": {}}

    sparse_mode = not dense_mode
    if sparse_mode:
        assert global_state is not None, "global_state is required in sparse mode"
        assert (byte_budgets is not None) or (fractions is not None)

    for cid in selected_clients:
        model = client_models[cid]
        optimizer = client_optimizers[cid]
        dl = client_dataloaders[cid]

        # Local training
        model.train()
        tot, nb = 0.0, 0
        examples = 0
        for _ in range(LOCAL_EPOCHS):
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                loss_i = criterion(out, y)
                loss_i.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tot += float(loss_i.item())
                nb += 1
                examples += y.size(0)

        if pbar is not None:
            pbar.update(1)

        train_losses[cid] = tot / max(1, nb)
        train_stats["steps_per_client"][cid] = nb
        train_stats["examples_per_client"][cid] = examples

        # Dense mode: send full model
        if dense_mode:
            client_updates[cid] = copy.deepcopy(model.state_dict())
            used_bytes_per_client[cid] = float(bytes_full)
            continue

        # Sparse mode
        local_sd = model.state_dict()

        # delta = local - global
        delta = {}
        for k in global_state.keys():
            lv = local_sd[k]
            gv = global_state[k]
            if (
                torch.is_tensor(lv)
                and torch.is_tensor(gv)
                and lv.is_floating_point()
                and gv.is_floating_point()
            ):
                delta[k] = lv - gv
            else:
                delta[k] = lv if torch.is_tensor(lv) else gv

        # determine per-client budget
        if byte_budgets is not None:
            byte_budget = int(byte_budgets.get(cid, 0))
        else:
            byte_budget = int(float(fractions.get(cid, 0.0)) * float(bytes_full))
        byte_budget = max(0, byte_budget)

        # compress delta (floats only)
        sparse_up, nnz = compress_delta_topk_by_bytes(
            delta_state=delta,
            byte_budget=byte_budget,
            value_bytes=VALUE_BYTES,
            index_bytes=INDEX_BYTES,
            min_keep=MIN_NNZ,
        )
        used_bytes = nnz * (VALUE_BYTES + INDEX_BYTES)
        used_bytes_per_client[cid] = float(used_bytes)

        # reconstruct û from sparse_up for densified payload
        u_hat = _sd_zero_like(global_state)
        for name, g_t in global_state.items():
            if (
                (name not in sparse_up)
                or (not torch.is_tensor(g_t))
                or (not g_t.is_floating_point())
            ):
                continue
            info = sparse_up[name]
            flat = u_hat[name].view(-1)
            idx = info["idx"].to(torch.long).to(flat.device)
            val = info["val"].to(g_t.dtype).to(flat.device)
            valid = (idx >= 0) & (idx < flat.numel())
            if not torch.all(valid):
                idx = idx[valid]
                val = val[valid]
            flat[idx] = val

        if densify_for_aggregator:
            dense_payload = {}
            for name, g_t in global_state.items():
                if not torch.is_tensor(g_t):
                    dense_payload[name] = g_t
                    continue
                if not g_t.is_floating_point():
                    dense_payload[name] = local_sd[name].clone()
                else:
                    dense_payload[name] = (g_t + u_hat[name]).to(g_t.dtype)
            client_updates[cid] = dense_payload
        else:
            client_updates[cid] = sparse_up

    round_used_bytes = sum(used_bytes_per_client.values())

    train_stats["total_steps"] = int(sum(train_stats["steps_per_client"].values()))
    train_stats["total_examples"] = int(
        sum(train_stats["examples_per_client"].values())
    )

    return (
        client_updates,
        train_losses,
        used_bytes_per_client,
        round_used_bytes,
        train_stats,
    )


def train_cifar10():

    # 1) build model
    global_model = CifarClientModel(num_classes=10).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # size-based constants using this model
    sd = global_model.state_dict()
    bytes_full = float(sum(
        t.numel() * t.element_size()
        for t in sd.values()
        if hasattr(t, "numel")
    ))

    # bits/entry for old accounting
    OLD_BITS_PER_ENTRY = compute_old_bits_per_entry(sd)

    # 2) data
    (
        clients_datasets,
        client_train_loaders,
        client_val_loaders,
        client_test_loaders,
        (x_test, y_test),
    ) = build_clients_from_cifar10_split(
        num_clients=NUM_CLIENTS,
        batch_size=BATCH_SIZE,
        val_split=VALIDATION_SPLIT,
        test_split=0.1,
        alpha=DIRICHLET_ALPHA,
        seed=0,
    )
    global_test_loader = get_global_test_loader(BATCH_SIZE)

    print_client_class_dist(clients_datasets)

    # 3) optimizers & aggregator
    client_models = {
        cid: copy.deepcopy(global_model).to(DEVICE)
        for cid in range(NUM_CLIENTS)
    }
    client_optimizers = {
        cid: torch.optim.SGD(
            client_models[cid].parameters(),
            lr=1e-4,
            momentum=0.9,
            weight_decay=1e-4,
        )
        for cid in range(NUM_CLIENTS)
    }
    server_aggregator = ServerAggregator(global_model, device=DEVICE)

    # 4) bandwidth setup
    B_bytes = float(BANDWIDTH_LIMIT) * bytes_full

    print("\n==== Bandwidth Reference (CIFAR-10) ====")
    print(f"[INIT] One full model: {_mb(bytes_full):.2f} MB")
    print(f"[INIT] Round bandwidth limit β={BANDWIDTH_LIMIT}: {_mb(B_bytes):.2f} MB")
    print("========================================\n")

    # Initial equal compression ratio across clients
    equal_fraction = B_bytes / (NUM_CLIENTS * bytes_full)
    equal_fraction = float(np.clip(equal_fraction, 0.0, 1.0))
    fractions_for_next_round = {cid: equal_fraction for cid in range(NUM_CLIENTS)}

    min_local_accuracies = []
    avg_local_accuracies = []

    for r in range(1, NUM_ROUNDS + 1):
        print(f"\n Round {r}/{NUM_ROUNDS}")

        fractions_for_round = fractions_for_next_round.copy()

        # Sync all clients
        clients_sync(global_model, client_models)

        # Per-client validation metrics (before training)
        all_val_acc = []
        val_losses_all = {}

        for client_id in client_models.keys():
            acc = test_accuracy(
                client_models[client_id], client_val_loaders[client_id], DEVICE
            )
            all_val_acc.append(acc)

            loss = calculate_loss(
                client_models[client_id],
                client_val_loaders[client_id],
                criterion,
                DEVICE,
            )
            val_losses_all[client_id] = loss

        min_acc = min(all_val_acc)
        avg_acc = sum(all_val_acc) / len(all_val_acc)
        min_local_accuracies.append(min_acc)
        avg_local_accuracies.append(avg_acc)
        print(f" Local Val Acc (Round {r}): Min={min_acc:.2f}%, Avg={avg_acc:.2f}%")

        grad_norms_all = None
        if ALLOCATION_METRIC == "grad_norm":
            grad_norms_all = {}
            for cid in client_models.keys():
                gn = estimate_grad_norm(
                    client_models[cid],
                    client_val_loaders[cid],
                    criterion,
                    DEVICE,
                    max_batches=1,
                )
                grad_norms_all[cid] = gn

        selected_clients = list(range(NUM_CLIENTS))

        with tqdm(total=len(selected_clients), desc=f"Training Clients (Round {r})", leave=True, dynamic_ncols=True) as pbar:
            global_state = copy.deepcopy(global_model.state_dict())
            (
                client_updates,
                train_losses,
                used_bytes_per_client,
                round_used_bytes,
                train_stats,
            ) = clients_update(
                selected_clients,
                client_models,
                client_optimizers,
                client_train_loaders,
                criterion,
                DEVICE,
                fractions=fractions_for_round,
                global_state=global_state,
                densify_for_aggregator=True,
                dense_mode=False,
                bytes_full=bytes_full,
                pbar=pbar,
            )

        print(f"\n[BW] Round used: {_mb(round_used_bytes):.2f} MB "
              f"(limit {_mb(B_bytes):.2f} MB)")

        # Aggregate
        server_aggregator.aggregate(client_updates, val_losses_all)

        # Global test accuracy
        global_test_acc = test_accuracy(global_model, global_test_loader, DEVICE)
        print(f"Global test acc: {global_test_acc:.2f}%")

        # Warm-up rounds with equal CR
        if r <= INITIAL_ROUNDS:
            fractions_for_next_round = {cid: equal_fraction for cid in range(NUM_CLIENTS)}
            continue

        # Compute scores for next-round allocation
        if ALLOCATION_METRIC == "grad_norm":
            score_dict = grad_norms_all
        elif ALLOCATION_METRIC == "val_loss":
            score_dict = val_losses_all
        else:
            raise ValueError(f"Unknown ALLOCATION_METRIC: {ALLOCATION_METRIC}")

        planned_bytes_init, _, _ = allocation_from_scores(
            score_dict=score_dict,
            B_bytes=B_bytes,
            bytes_full=bytes_full,
        )

        planned_per_client_bytes_final = []
        cache_hits = 0
        cache_misses = 0

        for i, cid in enumerate(selected_clients):
            B_i_bytes_raw = planned_bytes_init[i]

            info, hit = get_or_compute_compression(
                B_i_bytes_raw,
                global_model.state_dict(),
                OLD_BITS_PER_ENTRY,
                bytes_full,
            )
            if hit:
                cache_hits += 1
            else:
                cache_misses += 1

            planned_per_client_bytes_final.append(info["bytes"])

        print(f"[CT] cache hits={cache_hits}, cache misses={cache_misses}")

        # Re-scale to match total B_bytes
        sum_bytes = sum(planned_per_client_bytes_final)
        if sum_bytes > 0:
            scale = B_bytes / sum_bytes
            planned_per_client_bytes_final = [b * scale for b in planned_per_client_bytes_final]

        fractions_for_next_round = {
            cid: float(np.clip(planned_per_client_bytes_final[i] / bytes_full, 0.0, 1.0))
            for i, cid in enumerate(selected_clients)
        }



def train_utmobilenet(base_path: str):
    """
    FedBand training loop for UTMobileNetTraffic2021 dataset.

    Uses:
      - ResNet1D 1D model
      - build_utmobilenet_clients() for client loaders
      - identical FedBand bandwidth allocation & caching
      - aggregation weighted by validation loss
    """

    # 1. Build per-client dataloaders
    (
        train_loaders,
        val_loaders,
        global_test_loader,
        num_features,
        num_classes,
        label_mapping,
    ) = build_utmobilenet_clients(
        num_clients=NUM_CLIENTS,
        batch_size=BATCH_SIZE,
        alpha=DIRICHLET_ALPHA,
        base_path=base_path,
    )

    print_class_counts(train_loaders, val_loaders, global_test_loader)

    # 2. Build model + client copies
    global_model = ResNet1D(
        num_features=num_features,
        num_classes=num_classes
    ).to(DEVICE)


    criterion = nn.CrossEntropyLoss()

    sd = global_model.state_dict()
    bytes_full = float(sum(
        t.numel() * t.element_size()
        for t in sd.values()
        if hasattr(t, "numel")
    ))
    B_bytes = float(BANDWIDTH_LIMIT) * bytes_full
    OLD_BITS_PER_ENTRY = compute_old_bits_per_entry(sd)


    client_models = {
        cid: copy.deepcopy(global_model).to(DEVICE)
        for cid in range(NUM_CLIENTS)
    }

    client_optimizers = {
        cid: torch.optim.Adam(
            client_models[cid].parameters(),
            lr=1e-4,
            weight_decay=1e-4,
        )
        for cid in range(NUM_CLIENTS)
    }

    server_aggregator = ServerAggregator(global_model, device=DEVICE)


    print("\n==== Bandwidth Reference (UTMobileNet) ====")
    print(f"[INIT] One full model: {_mb(bytes_full):.2f} MB")
    print(f"[INIT] Round bandwidth limit β={BANDWIDTH_LIMIT}: {_mb(B_bytes):.2f} MB")
    print("==========================================\n")

    # equal CR for warm-up rounds
    equal_fraction = B_bytes / (NUM_CLIENTS * bytes_full)
    equal_fraction = float(np.clip(equal_fraction, 0.0, 1.0))

    fractions_for_next_round = {
        cid: equal_fraction for cid in range(NUM_CLIENTS)
    }

    # 4. Main FedBand loop
    for r in range(1, NUM_ROUNDS + 1):
        print(f"\n🔄 Round {r}/{NUM_ROUNDS}")

        fractions_for_round = fractions_for_next_round.copy()

        # Sync clients to server
        clients_sync(global_model, client_models)

        # Collect validation metrics before local training
        val_losses_all = {}
        val_accs_all = []

        for cid in range(NUM_CLIENTS):
            loss_val = calculate_loss(
                client_models[cid],
                val_loaders[cid],
                criterion,
                DEVICE,
            )
            val_losses_all[cid] = loss_val

            # we compute accuracy for monitoring only
            acc_val = test_accuracy(
                client_models[cid],
                val_loaders[cid],
                DEVICE,
            )
            val_accs_all.append(acc_val)

        print(
            f"Local Val Acc (Round {r}): "
            f"Min={min(val_accs_all):.2f}%, Avg={np.mean(val_accs_all):.2f}%"
        )

        # Optionally compute gradient norms
        grad_norms_all = None
        if ALLOCATION_METRIC == "grad_norm":
            grad_norms_all = {}
            for cid in range(NUM_CLIENTS):
                gn = estimate_grad_norm(
                    client_models[cid],
                    train_loaders[cid],
                    criterion,
                    DEVICE,
                    max_batches=1,
                )
                grad_norms_all[cid] = gn

        selected_clients = list(range(NUM_CLIENTS))

        # Client updates (local training + compression)
        with tqdm(total=len(selected_clients),
                  desc=f"Training Clients (Round {r})") as pbar:

            global_state = copy.deepcopy(global_model.state_dict())

            (
                client_updates,
                train_losses,
                used_bytes_per_client,
                round_used_bytes,
                train_stats,
            ) = clients_update(
                selected_clients,
                client_models,
                client_optimizers,
                train_loaders,
                criterion,
                DEVICE,
                fractions=fractions_for_round,
                global_state=global_state,
                densify_for_aggregator=True,
                dense_mode=False,
                bytes_full=bytes_full,  
                pbar=pbar,
            )

        print(f"\n[BW] Round used: {_mb(round_used_bytes):.2f} MB "
              f"(limit {_mb(B_bytes):.2f} MB)")

        # Aggregate weighted by validation loss
        server_aggregator.aggregate(client_updates, val_losses_all)

        # Evaluate global model
        global_test_acc = test_accuracy(global_model, global_test_loader, DEVICE)
        print(f" Global Test Accuracy: {global_test_acc:.2f}%")

        # Warm-up: use equal compression
        if r <= INITIAL_ROUNDS:
            fractions_for_next_round = {
                cid: equal_fraction for cid in selected_clients
            }
            continue

        # Compute next-round bandwidth fractions
        if ALLOCATION_METRIC == "grad_norm":
            score_dict = grad_norms_all
        elif ALLOCATION_METRIC == "val_loss":
            score_dict = val_losses_all
        else:
            raise ValueError(f"Unknown ALLOCATION_METRIC: {ALLOCATION_METRIC}")

        planned_bytes_init, _, _ = allocation_from_scores(
            score_dict=score_dict,
            B_bytes=B_bytes,
            bytes_full=bytes_full,
        )

        planned_per_client_bytes_final = []
        cache_hits = 0
        cache_misses = 0

        for cid, planned_bytes in enumerate(planned_bytes_init):
            info, hit = get_or_compute_compression(
                planned_bytes,
                global_model.state_dict(),
                OLD_BITS_PER_ENTRY,
                bytes_full,
            )
            if hit:
                cache_hits += 1
            else:
                cache_misses += 1

            planned_per_client_bytes_final.append(info["bytes"])

        print(f"[CT] cache hits={cache_hits}, cache misses={cache_misses}")

        # Rescale to exactly match the round budget
        S = sum(planned_per_client_bytes_final)
        if S > 0:
            scale = B_bytes / S
            planned_per_client_bytes_final = [
                b * scale for b in planned_per_client_bytes_final
            ]

        fractions_for_next_round = {
            cid: float(
                np.clip(planned_per_client_bytes_final[cid] / bytes_full, 0.0, 1.0)
            )
            for cid in selected_clients
        }

