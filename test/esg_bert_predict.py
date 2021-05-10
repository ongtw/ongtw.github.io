import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from time import perf_counter

#
# Global Defs
#
# ESG-BERT's default label mapping
esg_bert_label_map = {
    0: "__label__Business_Ethics",
    1: "__label__Data_Security",
    2: "__label__Access_And_Affordability",
    3: "__label__Business_Model_Resilience",
    4: "__label__Competitive_Behavior",
    5: "__label__Critical_Incident_Risk_Management",
    6: "__label__Customer_Welfare",
    7: "__label__Director_Removal",
    8: "__label__Employee_Engagement_Inclusion_And_Diversity",
    9: "__label__Employee_Health_And_Safety",
    10: "__label__Human_Rights_And_Community_Relations",
    11: "__label__Labor_Practices",
    12: "__label__Management_Of_Legal_And_Regulatory_Framework",
    13: "__label__Physical_Impacts_Of_Climate_Change",
    14: "__label__Product_Quality_And_Safety",
    15: "__label__Product_Design_And_Lifecycle_Management",
    16: "__label__Selling_Practices_And_Product_Labeling",
    17: "__label__Supply_Chain_Management",
    18: "__label__Systemic_Risk_Management",
    19: "__label__Waste_And_Hazardous_Materials_Management",
    20: "__label__Water_And_Wastewater_Management",
    21: "__label__Air_Quality",
    22: "__label__Customer_Privacy",
    23: "__label__Ecological_Impacts",
    24: "__label__Energy_Management",
    25: "__label__GHG_Emissions",
}
esg_bert_ubs_label_map = {
    0: "Governance",
    1: "Governance",
    2: "Products and Services",
    3: "Governance",
    4: "People",
    5: "Governance",
    6: "People",
    7: "Governance",
    8: "People",
    9: "People",
    10: "People",
    11: "People",
    12: "Governance",
    13: "Climate Change",
    14: "Products and Services",
    15: "Products and Services",
    16: "Products and Services",
    17: "Products and Services",
    18: "Governance",
    19: "Pollution and Waste",
    20: "Water",
    21: "Pollution and Waste",
    22: "People",
    23: "Climate Change",
    24: "Climate Change",
    25: "Climate Change",
}
model_path = "model/"  # folder where ESG-BERT model is stored
device = torch.device("cpu")  # default compute device
topK = 3  # top-K results we are interested in


def time_it(func):
    """Decorator to time function

    Args:
        func ([type]): function to be timed
    """

    def wrapper(*args, **kwargs):
        stime = perf_counter()
        res = func(*args, **kwargs)
        etime = perf_counter()
        dur = etime - stime
        print(f"  {func.__name__}() time={dur:.0f}s")
        return res, dur

    return wrapper


def softmax(x):
    """
    The standard softmax function
    Use "axis=-1, keepdims=True" to eval softmax row-by-row,
    not flattening the input
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    smax = e_x / e_x.sum(axis=-1, keepdims=True)
    return smax


def get_data(filename: str) -> pd.DataFrame:
    """Return dataframe from given filename

    Args:
        filename (str): input filename

    Returns:
        pd.DataFrame: the return dataframe
    """
    df = pd.read_table(filename)
    df = df.dropna()
    return df


def init_model():
    """Initialise and return ESG-BERT model

    Returns:
        [type]: ESG-BERT model
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=26,
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to(device)
    return model


def get_input_params():
    """Return user inputs on CLI

    Returns:
        [object]: args object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="Input csv file", type=str, required=True)
    parser.add_argument(
        "-o",
        type=str,
        default="output.csv",
        help="Output file name (default: output.csv)",
    )
    parser.add_argument(
        "-v",
        type=int,
        default=1,
        help="Verbose setting {0 (quiet), 1 (head), 2 (show all)} (default: 0)",
    )
    parser.add_argument(
        "-b",
        type=int,
        default=30,
        help="Batch size for prediction (default: 30)",
    )
    args = parser.parse_args()
    return args


# For displaying dataframe contents
line_counter = 0


def print_line(x):
    """Helper function for printing dataframe contents

    Args:
        x ([type]): a dataframe row
    """
    global line_counter
    line_counter += 1
    print(f"{line_counter}. {x[0]}")


@time_it
def predict(df: pd.DataFrame, model, tokenizer, device):
    """Use model and tokenizer to predict given dataframe of text

    Args:
        df (pd.DataFrame): input text
        model ([type]): ESG-BERT model
        tokenizer ([type]): ESG-BERT tokenizer
        device ([type]): PyTorch device

    Returns:
        [type]: ESG-BERT transformer predictions
    """
    # NB: inputs columns are ["input_ids", "token_type_ids", "attention_mask"]
    inputs = tokenizer(
        df.iloc[:, 0].tolist(),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    predictions = model(
        inputs["input_ids"].to(device),
        token_type_ids=inputs["token_type_ids"].to(device),
    )
    return predictions


if __name__ == "__main__":
    args = get_input_params()
    csv_fullpath = args.f
    output_filename = args.o
    verbose = args.v
    batch_size = args.b
    df = get_data(csv_fullpath)
    n = len(df)
    if verbose > 0:
        n = len(df)
        print("--- input ---")
        if verbose == 2:
            df.apply(print_line, axis=1)
        else:
            df.head().apply(print_line, axis=1)
        print("--- end ---")

    prog_stime = perf_counter()
    total_model_dur = 0

    # init model and tokenizer
    model = init_model()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"n={n}, batch_size={batch_size}")
    df_grp_obj = df.groupby(np.arange(n) // batch_size)
    num_grps = len(df_grp_obj)

    # predict
    for k, g in df_grp_obj:
        print(f"Predicting Batch {1+k}/{num_grps}:")
        print(g.index[0], g.head(1).iloc[0][0])
        print("...")
        print(g.index[-1], g.tail(1).iloc[0][0])
        predictions, model_dur = predict(g, model, tokenizer, device)
        total_model_dur += model_dur
        print()

        # convert model predictions to probabilities
        preds = predictions.logits.detach().numpy()
        preds_probs = softmax(preds)
        # find top-K indices
        preds_idx_sorted = np.argsort(-preds)  # -ve to reverse array
        preds_idx_topK = preds_idx_sorted[:, :topK]
        # find top-K probabilities
        preds_probs_topK_idx = (np.arange(len(g))[:, None], preds_idx_topK)
        preds_probs_topK = preds_probs[preds_probs_topK_idx]
        # find top-K ESG-BERT labels
        esg_bert_label_np = np.array(list(esg_bert_label_map.items()))[:, 1]
        preds_labels_topK = esg_bert_label_np[preds_idx_topK]
        # then find the corresponding top-K UBS SI labels
        ubs_label_np = np.array(list(esg_bert_ubs_label_map.items()))[:, 1]
        ubs_labels_topK = ubs_label_np[preds_idx_topK]
        # altogether in one big dataframe: "We are family, I got all my sisters with me!"
        g["ESG_BERT_Labels"] = preds_labels_topK[:, 0]
        g["UBS_SI_Labels"] = ubs_labels_topK[:, 0]
        g["Probs"] = preds_probs_topK[:, 0]
        g.columns = ["Sentences", "ESG_BERT_Labels", "UBS_SI_Labels", "Prob"]
        # write out to csv directly, don't need to keep in memory
        if k == 0:
            g.to_csv(output_filename)  # overwrite file on first batch
        else:
            g.to_csv(output_filename, mode="a", header=False)

    print(f"Total model prediction time={total_model_dur:.0f}s")

    prog_etime = perf_counter()
    prog_dur = prog_etime - prog_stime
    print(f"Total time taken={prog_dur:.0f}s")
