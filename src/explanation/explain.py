def extract_explanation(explanation_dict):
    h = explanation_dict["concepts"]
    theta = explanation_dict["relevances"]
    return list(zip(h.detach().cpu(), theta.detach().cpu()))
