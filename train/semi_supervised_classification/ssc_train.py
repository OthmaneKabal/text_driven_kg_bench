import torch
import torch.nn as nn
import torch.optim as optim
from utilities import set_seed
from evaluate.evaluate_ssc import ssc_evaluate_split
## MINI batch Training
def batch_train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    mask_input_id = torch.isin(data.n_id, data.input_id)
    mask = mask_input_id & data.train_mask & (data.y >= 0)
    loss = criterion(out[mask], data.y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def training_loop_minibatch(model, train_loader, val_loader, test_loader, config, epochs=100, seed=42, top_k=None):
    set_seed(seed)
    device = torch.device(config["device"])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=config.get("lr", 0.001),
                           weight_decay=config.get("weight_decay", 5e-4))
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    best_val_f1 = 0.0
    best_val_metrics = {}
    best_model_state = None

    print("Starting training...")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        # -------- TRAINING --------
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            batch_loss = batch_train(model, batch, optimizer, criterion)
            total_loss += batch_loss

        avg_loss = total_loss / len(train_loader)

        # -------- EVALUATION --------
        train_metrics = ssc_evaluate_split(model, train_loader, "train", config)
        val_metrics   = ssc_evaluate_split(model, val_loader, "val", config)
        test_metrics  = ssc_evaluate_split(model, test_loader, "test", config)

        # Sélection du meilleur modèle basé sur F1 validation
        if val_metrics and val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_val_metrics = val_metrics
            best_model_state = model.state_dict()
            # save_best_classifier_and_config(model, config, best_val_metrics, top_k=top_k)

        # -------- LOGGING --------
        print(f"Epoch {epoch:3d}/{epochs}")
        print(f"  Loss:     {avg_loss:.4f}")
        print(f"  Train -   Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f}")
        print(f"  Val   -   Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
        print(f"  Test  -   Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f}")
        print(f"  Best Val F1: {best_val_f1:.4f}")
        print("-" * 60)

    # -------- FIN : TEST avec le meilleur modèle --------
    print(f"\nTraining completed! Best Val F1: {best_val_f1:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)  # restaurer le meilleur modèle
        final_test_metrics = ssc_evaluate_split(model, test_loader, "test", config)
    else:
        final_test_metrics = {"accuracy": 0.0, "f1": 0.0, "recall": 0.0, "precision": 0.0}

    return {
        "best_val": best_val_metrics,
        "final_test": final_test_metrics
    }
