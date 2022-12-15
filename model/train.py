def train_model(num_epochs, model, device, loader, optimizer, loss_fn):
    # Set the model to training mode
    model.train(mode=True)
    for epoch in range(num_epochs):
        # Loop over the dataloader to get the training batches
        for data, labels in loader:
            # Move the data and labels to the appropriate device (e.g. GPU)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass through the model
            outputs = model(data)

            # Compute the loss
            loss = loss_fn(outputs, labels)

            # Backward pass through the model to compute the gradients
            loss.backward()

            # Update the model parameters using the optimizer
            optimizer.step()

        # Print the loss for every n epochs
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

    model.train(mode=False)
