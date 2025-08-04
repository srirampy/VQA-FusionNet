import torch
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device="cuda", best_model_name="vqa_model"):
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        train_loop = tqdm(train_loader, desc="Training", leave=False)
        for img_features, text_features, labels in train_loop:
            img_features, text_features, labels = img_features.to(device), text_features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(img_features, text_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            train_loop.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        val_loop = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for img_features, text_features, labels in val_loop:
                img_features, text_features, labels = img_features.to(device), text_features.to(device), labels.to(device)
                outputs = model(img_features, text_features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                val_loop.set_postfix(loss=loss.item())

        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"outputs/best_{best_model_name.replace('/', '_')}.pth")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
