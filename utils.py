import torch


def add_noise(input, mean=0, std=1, propogation=0.1):
    noise = torch.normal(mean=mean, std=std, size=input.shape, device=input.device)
    output = input + propogation * noise

    return output


def train(
    model,
    train_loader,
    optimizer,
    criterion,
    epochs,
    device,
    save_dir,
    save_interval=20,
    log_interval=10,
):
    print("Training started...")
    losses = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()

            pure = data.to(device)
            noisy = add_noise(pure)
            denoised = model(noisy)

            loss = criterion(denoised, pure)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if batch_idx % log_interval == 0:
                print(
                    f"Epoch: {epoch}\t"
                    f"[{batch_idx * len(data)}/{len(train_loader.dataset)}"
                    f" ({100. * batch_idx / len(train_loader):.0f}%)]\t"
                    f"Loss: {loss.item() / len(data):.6f}"
                )

        losses.append(epoch_loss / len(train_loader.dataset))
        if epoch % save_interval == 0:
            torch.save(
                model.state_dict(),
                save_dir / f"autoencodercnn_epoch_{epoch}.pth",
            )

    return losses


def test(model, test_loader, criterion, device):
    model.eval()

    test_loss = 0
    noisy_imgs = []
    denoised_imgs = []

    with torch.no_grad():
        for data, _ in test_loader:
            pure = data.to(device)
            noisy = add_noise(pure)
            denoised = model(noisy)

            loss = criterion(denoised, pure)
            test_loss += loss.item()

            for n, d in zip(noisy, denoised):
                noisy_imgs.append(n.cpu())
                denoised_imgs.append(d.cpu())

    test_loss /= len(test_loader.dataset)
    print(f"Average test loss: {test_loss:.6f}")

    return noisy_imgs, denoised_imgs
