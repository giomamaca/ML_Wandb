import torch
import torch.nn as nn
import wandb

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss_sum += criterion(out, y).item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
    return loss_sum / n, correct / n

def train(model, train_loader, val_loader, config, project="fer2013-experiments", run_name=None):
    run = wandb.init(project=project, name=run_name, config=config, reinit=True)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt_cls = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}[config.get('optimizer', 'adam')]
    kwargs = {'momentum': 0.9} if config.get('optimizer') == 'sgd' else {}
    optimizer = opt_cls(model.parameters(), lr=config['lr'], **kwargs)

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

        train_loss, train_acc = loss_sum / n, correct / n
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                   'val_loss': val_loss, 'val_acc': val_acc})
        print(f"epoch {epoch:2d} | train {train_loss:.3f}/{train_acc:.3f} | val {val_loss:.3f}/{val_acc:.3f}")

    run.finish()
    return model
