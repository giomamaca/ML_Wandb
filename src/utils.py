import math, torch, torch.nn as nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def check_initial_loss(model, loader, num_classes=7):
    model = model.to(DEVICE).eval()
    x, y = next(iter(loader))
    with torch.no_grad():
        loss = nn.CrossEntropyLoss()(model(x.to(DEVICE)), y.to(DEVICE)).item()
    expected = math.log(num_classes)
    print(f"initial loss = {loss:.3f}, expected ~{expected:.3f}")
    return abs(loss - expected) < 0.3

def overfit_small_batch(model, loader, steps=200, lr=1e-3):
    model = model.to(DEVICE).train()
    x, y = next(iter(loader))
    x, y = x[:64].to(DEVICE), y[:64].to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for step in range(steps):
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            acc = (out.argmax(1) == y).float().mean().item()
            print(f"step {step:3d} | loss {loss.item():.4f} | acc {acc:.3f}")

def check_gradients(model, loader):
    model = model.to(DEVICE).train()
    x, y = next(iter(loader))
    nn.CrossEntropyLoss()(model(x.to(DEVICE)), y.to(DEVICE)).backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            print(f"{name:40s} grad norm = {p.grad.norm().item():.6f}")
    model.zero_grad()
