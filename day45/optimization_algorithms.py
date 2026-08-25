import torch


def objective(x):
    return 0.1 * x[0] ** 2 + 2 * x[1] ** 2


def optimize(optimizer_class, steps=50, **kwargs):
    parameter = torch.nn.Parameter(torch.tensor([-5.0, -2.0]))
    optimizer = optimizer_class([parameter], **kwargs)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = objective(parameter)
        loss.backward()
        optimizer.step()
    return parameter.detach(), objective(parameter).item()


if __name__ == "__main__":
    settings = {
        "SGD": (torch.optim.SGD, {"lr": 0.1}),
        "Momentum": (torch.optim.SGD, {"lr": 0.05, "momentum": 0.9}),
        "AdaGrad": (torch.optim.Adagrad, {"lr": 0.5}),
        "RMSProp": (torch.optim.RMSprop, {"lr": 0.05}),
        "Adam": (torch.optim.Adam, {"lr": 0.2}),
    }
    for name, (optimizer, options) in settings.items():
        point, loss = optimize(optimizer, **options)
        print(f"{name:8s} point={point.tolist()}, loss={loss:.6f}")
