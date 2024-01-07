# import mlflow
#
# relative_path_to_model = {
#    'NN_PBE': '583612537407811688/d47165289483405692b8424df7009906/artifacts/PBE_8_32_0.4_final',
#    'NN_XALPHA': '429050431669758440/3886423064a64fc199d7879c234fa63c/artifacts/XALPHA_32_32_0.4_final'
# }
# with mlflow.start_run():
#    for model_name in 'NN_PBE', 'NN_XALPHA':
#        model_path = 'mlruns/' + relative_path_to_model[model_name]
#        model = mlflow.pytorch.load_model(model_path)
#        mlflow.pytorch.log_state_dict(model.state_dict(), model_name)
#        state_dict_uri = mlflow.get_artifact_uri(model_name)
#        print(model_name)
#        print(state_dict_uri)
#        print('\n\n')
from prepare_data import prepare

prepare()
