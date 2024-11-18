import torch
def test_epoch(data_loader, model, mae, rmse, r2, device):
    model.eval()
    model.to(device)
    mae_list = []
    rmse_list = []
    r2_list = []
    with torch.no_grad():  
        for i, (input, target,_) in enumerate(data_loader):
            input_var = (input[0].to(device, non_blocking=True),
                     input[1].to(device, non_blocking=True),
                     input[2].to(device, non_blocking=True),
                     [crys_idx.to(device, non_blocking=True) for crys_idx in input[3]])
            target_var = target.to(device, non_blocking=True)
            output = model(*input_var)
            # calculate the mae_loss
            mae_a = mae(output, target_var).item()
            mae_list.append(mae_a)
            # calculate the rmse_loss
            rmse_a = rmse(output, target_var).item()
            rmse_list.append(rmse_a)
            # calculate the r2_score
            r2_a = r2(output, target_var).item()
            r2_list.append(r2_a)

        a = torch.tensor(mae_list, dtype=torch.float32).mean().item()
        b = torch.tensor(rmse_list, dtype=torch.float32).mean().item()
        c = torch.tensor(r2_list, dtype=torch.float32).mean().item()

    return a, b, c

def test_all_dataset(data_loader, model, device=torch.device('cuda')):
    y_pred=[]
    y_truth=[]
    cif_ids = []
    model.eval()
    model.to(device)
    for i, (input, target, cif) in enumerate(data_loader):
        input_var = (input[0].to(device, non_blocking=True),
                            input[1].to(device, non_blocking=True),
                            input[2].to(device, non_blocking=True),
                            [crys_idx.to(device, non_blocking=True) for crys_idx in input[3]])
        pred, _ = model(*input_var)
        y_pred.extend(pred)
        y_truth.extend(target)
        cif_ids.extend(cif)

    #y_pred_cpu = [pred.detach().cpu().item() for pred in y_pred]
    #y_truth_cpu = [truth.detach().cpu().item() for truth in y_truth]

    return y_pred, y_truth, cif_ids