import torch


def train_epoch(data_loader, model, criterion, optimizer, device, training=True):
    loss_list = []
    model.to(device)
    if training == True:
        model.train()
        for i, (input, target,_) in enumerate(data_loader):
            # Transfer the data to GPU 
            input_var = (input[0].to(device, non_blocking=True),
                        input[1].to(device, non_blocking=True),
                        input[2].to(device, non_blocking=True),
                        [crys_idx.to(device, non_blocking=True) for crys_idx in input[3]])
            target_var = target.to(device, non_blocking=True)
            
            optimizer.zero_grad()  # Clear gradients.
            outputs = model(*input_var)  # Perform a single forward pass.
            loss = criterion(outputs, target_var)  # Compute the loss 
            loss_list.append(loss.item())
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
    else:
        model.eval()
        with torch.no_grad():
            for i, (input, target,_) in enumerate(data_loader):
                input_var = (input[0].to(device, non_blocking=True),
                     input[1].to(device, non_blocking=True),
                     input[2].to(device, non_blocking=True),
                     [crys_idx.to(device, non_blocking=True) for crys_idx in input[3]])
                target_var = target.to(device, non_blocking=True)
                outputs = model(*input_var)
                loss = criterion(outputs, target_var)
                loss_list.append(loss.item())
    
    return torch.tensor(loss_list, dtype=torch.float32).mean()
