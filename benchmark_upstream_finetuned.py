import torch
import warnings
import json
import torch.nn.functional as F

from ssl_wearables.sslearning.models.accNet import Resnet, DropoutHead
warnings.filterwarnings("ignore")
device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )


def mc_dropout_finetuned_inference(model, X):
    
    # Everything stays in eval mode.
    model.eval()

    predictions = []
    
    x = torch.from_numpy(
        np.asarray(X)
    ).float().to(device)
    
    feats = model.feature_extractor(x)
    
    feats = feats.view(
            feats.shape[0],
            -1
        )
    
    for _ in range(50):
        
        output = model.classifier(feats,mc_dropout=True)

        # Explicitly activate dropout exactly as in training evaluation

        mu = output[:, 0]

        predictions.append(mu)

    predictions = torch.stack(
        predictions,
        dim=0,
    )

    mu = predictions.mean(
        dim=0,
    )

    var = predictions.var(
        dim=0,
        unbiased=False,
    )

    return feats.detach().cpu().numpy(), mu.detach().cpu().numpy(), var.detach().cpu().numpy()
    
def load_HR_encoder(ckpt_path = "ssl_wearables/output_hr/HR_reg_gaussian_oxfordhub_25_final.mdl",
output_size = 2, is_mc_dropout=False):

    downstream_model = Resnet(
        output_size=output_size,
        resnet_version=1,
        epoch_len=10,
            is_eva=True,         # <<<<<< IMPORTANT
    is_mtl=False,
    ).to(device)
    
    pretrained = torch.load(
        ckpt_path,
        map_location=device,
        )

        # Remove toda a head usada durante o pretraining
    if is_mc_dropout:
        downstream_model.classifier = DropoutHead(
                output_size=1
            ).to(device)
    # Mantém a nova head e atualiza o restante com os pesos pretrained
    model_dict = downstream_model.state_dict()
    
    model_dict.update({
        key: value
        for key, value in pretrained.items()
    })
    
    downstream_model.load_state_dict(model_dict)
    
    return downstream_model


GAUSSIAN_LOG_VAR_MIN = -10.0
GAUSSIAN_LOG_VAR_MAX = 10.0

def gaussian_regression_inference(model, X):

    model.eval()

    x = torch.from_numpy(
        np.asarray(X)
    ).float().to(device)

    with torch.no_grad():

        # ==========================================
        # YOUR HR-FINE-TUNED BACKBONE
        # ==========================================

        features = model.feature_extractor(x)

        features = features.view(
            features.shape[0],
            -1
        )

        # ==========================================
        # YOUR TRAINED GAUSSIAN HEAD
        # ==========================================

        raw_output = model.classifier(features)

        # HR prediction
        mu = raw_output[:, 0]

        # Learned log variance
        log_var = raw_output[:, 1].clamp(
            min=GAUSSIAN_LOG_VAR_MIN,
            max=GAUSSIAN_LOG_VAR_MAX
        )

        # Learned HR uncertainty
        var = torch.exp(
            log_var
        )


    return (
        features.cpu().numpy(),
        mu.cpu().numpy(),
        var.cpu().numpy()
    )



def student_t_head_inference(model, X):

    model.eval()

    x = torch.from_numpy(
        np.asarray(X)
    ).float().to(device)
    
    with torch.no_grad():

        # ==========================================
        # YOUR HR-FINE-TUNED BACKBONE
        # ==========================================

        features = model.feature_extractor(x)

        features = features.view(
            features.shape[0],
            -1
        )

        # ==========================================

        output = model.classifier(features)
 


        mu = output[:, 0]
    
        scale = (
            F.softplus(output[:, 1])
            + 1e-3
        )
    
        df = (
            2.01
            + F.softplus(output[:, 2])
        )
    
        # Student-t predictive variance
    #
    # Var = scale² * df / (df - 2)

        var = (
        scale.pow(2)
        * df
        / (df - 2.0)
    )

    return (features.cpu().numpy(),
        mu.cpu().numpy(),
        var.cpu().numpy()
    )


import pandas as pd
import json
import numpy as np

def take_id(path):
    patten = '/home/jovyan/'
    return path.replace(patten, '')

def make_upstream_results(output_size, encoder_path):

    path = "ssl_wearables/data/ssl/ssl_ai4wellness/data/test/file_list.csv"
    df_list = pd.read_csv(path)
    
    users = df_list.file_list.apply(take_id).unique()
    

    dfs = []

    with open("user_stats.json", "r") as file:
            user_stats = json.load(file)

    is_mc_dropout = True if output_size ==1 else False
    encoder = load_HR_encoder(output_size = output_size, ckpt_path = encoder_path, is_mc_dropout = is_mc_dropout)

    for user_path in users:

        data = np.load(user_path)
        y, X = data['hr'],  data['X']
        
        # X: (2268, 3, 300) in g units

        if output_size >2:
            
            features, predicted_hr, hr_uncertainty = (
                    student_t_head_inference(
                        encoder,
                        X
                    )
                )
        elif output_size == 2:
            
            features, predicted_hr, hr_uncertainty = (
                    gaussian_regression_inference(
                        encoder,
                        X
                    )
                )
        else:
            
            features, predicted_hr, hr_uncertainty =  mc_dropout_finetuned_inference(encoder, X)

        df_user = pd.DataFrame(features)
        user_id  = user_path.replace('ssl_wearables/data/ssl/ssl_ai4wellness/data/test/user_' , '').replace('.npz', '')
        user_stat = user_stats[user_id]

        # uncertainty learned during upstream HR task
        df_user["hr_uncertainty"] = hr_uncertainty

        # optional useful ablation variable
        df_user["predicted_hr"] = predicted_hr
        df_user['y'] = (y - user_stat['mean'])/user_stat['std']
        df_user['original'] = y
        # HAR task
        df_user["user_id"] = user_id
    
        dfs.append(df_user)
        
    df_final = pd.concat(dfs,ignore_index=True
)
    return df_final

def make_upstream_representation(encoder_head_output, encoder_paths):
    
    """ 
    encoder_head_output: 1 = mc_dropout
                      2 = gaussian
                      3 = student_t
                      
    encoder_paths: list[str]
    
    """
    
    is_mc_dropout = True if encoder_head_output ==1 else False
    encoder = load_HR_encoder(output_size = output_size, ckpt_path = encoder_path, is_mc_dropout = is_mc_dropout)

    for user_id, path in enumerate(encoder_paths):
        
        data = np.load(path)
        y, X = data['y'],  data['X']
        
        if encoder_head_output ==3:
            
            features, predicted_hr, hr_uncertainty = (
                    student_t_head_inference(
                        encoder,
                        X
                    )
                )
        elif encoder_head_output == 2:
            
            features, predicted_hr, hr_uncertainty = (
                    gaussian_regression_inference(
                        encoder,
                        X
                    )
                )
        else:
            features, predicted_hr, hr_uncertainty =  mc_dropout_finetuned_inference(encoder, X)

        df_user = pd.DataFrame(features)

        # uncertainty learned during upstream HR task
        df_user["u"] = hr_uncertainty

        df_user["user_id"] = user_id

        dfs.append(df_user)
        
    df_final = pd.concat(dfs,ignore_index=True
)
    return df_final
