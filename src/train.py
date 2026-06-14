import os
import torch
import torch.nn as nn
import numpy as np
import wandb

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMOTIONS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

def evaluate(model, loader, criterion, return_preds=False):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    preds_all, y_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss_sum += criterion(out, y).item() * y.size(0)
            p = out.argmax(1)
            correct += (p == y).sum().item()
            n += y.size(0)
            if return_preds:
                preds_all.append(p.cpu()); y_all.append(y.cpu())
    if return_preds:
        return loss_sum/n, correct/n, torch.cat(preds_all).numpy(), torch.cat(y_all).numpy()
    return loss_sum/n, correct/n

def train(model, train_loader, val_loader, config, project="fer2013-experiments",
          run_name=None, test_loader=None):
    run = wandb.init(project=project, name=run_name, config=config, reinit=True)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt_cls = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}[config.get('optimizer','adam')]
    kwargs = {'momentum': 0.9} if config.get('optimizer') == 'sgd' else {}
    optimizer = opt_cls(model.parameters(), lr=config['lr'],
                        weight_decay=config.get('weight_decay', 0.0), **kwargs)

    os.makedirs('checkpoints', exist_ok=True)
    ckpt_path = f"checkpoints/{run_name}.pt"
    best_val_acc = 0.0

    for epoch in range(config['epochs']):
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)

        train_loss, train_acc = loss_sum/n, correct/n
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                   'val_loss': val_loss, 'val_acc': val_acc,
                   'overfit_gap': train_acc - val_acc})
        print(f"epoch {epoch:2d} | train {train_loss:.3f}/{train_acc:.3f} | "
              f"val {val_loss:.3f}/{val_acc:.3f} | gap {train_acc-val_acc:+.3f}")

        if val_acc > best_val_acc:                       # best model -> disk
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)

    wandb.summary['best_val_acc'] = best_val_acc

    # reload best weights, final eval (PrivateTest if given, else val)
    model.load_state_dict(torch.load(ckpt_path))
    eval_loader = test_loader if test_loader is not None else val_loader
    split = 'test' if test_loader is not None else 'val'
    _, acc, preds, ys = evaluate(model, eval_loader, criterion, return_preds=True)
    wandb.summary[f'{split}_acc_best'] = acc

    wandb.log({'confusion_matrix': wandb.plot.confusion_matrix(
        preds=preds.tolist(), y_true=ys.tolist(), class_names=EMOTIONS)})

    cm = np.zeros((7,7), dtype=int)                      # per-class accuracy
    for t,p in zip(ys, preds): cm[t,p] += 1
    for i,name in enumerate(EMOTIONS):
        denom = cm[i].sum()
        wandb.summary[f'acc_{name}'] = float(cm[i,i]/denom) if denom else 0.0

    run.finish()
    return model, best_val_acc
