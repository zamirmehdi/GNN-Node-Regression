import time

import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.nn import GATConv, GATv2Conv, GraphConv, GCN2Conv
from dgl.nn.pytorch import GATConv, GATv2Conv, GCN2Conv, SAGEConv

import numpy as np
import pandas as pd
import json
import networkx as nx
from torchmetrics.regression import MeanSquaredLogError, MeanAbsolutePercentageError, MeanAbsoluteError

import matplotlib

matplotlib.use('Agg')  # Use the 'Agg' backend
matplotlib.use('TkAgg')  # Use the 'Agg' backend
import matplotlib.pyplot as plt

from models import *

dataset_files = {
    'dataset_files_chameleon': {
        'target_csv_file': 'wikipedia/chameleon/musae_chameleon_target.csv',
        'edges_csv_file': 'wikipedia/chameleon/musae_chameleon_edges.csv',
        'features_json_file': 'wikipedia/chameleon/musae_chameleon_features.json'
    },
    'dataset_files_squirrel': {
        'target_csv_file': 'wikipedia/squirrel/musae_squirrel_target.csv',
        'edges_csv_file': 'wikipedia/squirrel/musae_squirrel_edges.csv',
        'features_json_file': 'wikipedia/squirrel/musae_squirrel_features.json'
    },
    'dataset_files_crocodile': {
        'target_csv_file': 'wikipedia/crocodile/musae_crocodile_target.csv',
        'edges_csv_file': 'wikipedia/crocodile/musae_crocodile_edges.csv',
        'features_json_file': 'wikipedia/crocodile/musae_crocodile_features.json'
    }
}


def read_files(src):
    edges_df = pd.read_csv(src['edges_csv_file'])
    target_df = pd.read_csv(src['target_csv_file'])
    with open(src['features_json_file'], 'r') as json_file:
        node_features = json.load(json_file)
    return edges_df, node_features, target_df


def find_inlier_indexes(target_df):
    # print('\ntarget values distribution:\n',target_df['target'].describe())

    # Calculate Q1 and Q3
    Q1 = target_df['target'].quantile(0.25)
    Q3 = target_df['target'].quantile(0.75)
    # print('Qs:', Q1, Q3)

    # Calculate IQR
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # print('bounds:', lower_bound, upper_bound)

    # Filter and remove outliers
    df_filtered = target_df[(target_df['target'] >= lower_bound) & (target_df['target'] <= upper_bound)]
    indexes_to_keep = df_filtered['id'].tolist()

    # Outliers
    print(target_df[(target_df['target'] < lower_bound) | (target_df['target'] > upper_bound)])

    return indexes_to_keep


def filter_target_df(target_df, indexes_to_keep):
    filtered_target_df = target_df.copy()

    # Filter rows where 'id' is in the list
    filtered_target_df = filtered_target_df[filtered_target_df['id'].isin(indexes_to_keep)]

    # Create a mapping of old indexes to new indexes
    new_index_mapping = {old_index: new_index for new_index, old_index in enumerate(indexes_to_keep)}

    # Update 'id' column with new indexes
    filtered_target_df['id'] = filtered_target_df['id'].map(new_index_mapping)

    # Reset the index of the DataFrame
    filtered_target_df.reset_index(drop=True, inplace=True)

    return filtered_target_df, new_index_mapping


def filter_edges_df(edges_df, indexes_to_keep, new_index_mapping):
    # Create a boolean mask for filtering based on both 'id1' and 'id2'
    mask = edges_df['id1'].isin(indexes_to_keep) & edges_df['id2'].isin(indexes_to_keep)

    # Filter rows based on the mask
    filtered_edges_df = edges_df[mask].copy()  # Create a copy of the filtered DataFrame

    # Update 'id1' and 'id2' columns with new indexes
    filtered_edges_df['id1'] = filtered_edges_df['id1'].map(new_index_mapping)
    filtered_edges_df['id2'] = filtered_edges_df['id2'].map(new_index_mapping)

    # Reset the index of the DataFrame
    filtered_edges_df.reset_index(drop=True, inplace=True)

    return filtered_edges_df


def filter_node_features(node_features_dict, new_index_mapping, indexes_to_keep):
    # Filter and update the node_features dictionary without loops
    filtered_node_features = {str(new_index_mapping[int(old_index)]): features for old_index, features in
                              node_features_dict.items() if
                              int(old_index) in indexes_to_keep}

    return filtered_node_features


def remove_outliers(target_df, edges_df, node_features_dict):
    print('\nOutliers to Remove:')
    indexes_to_keep = find_inlier_indexes(target_df)

    target_df_filtered, new_index_mapping = filter_target_df(target_df, indexes_to_keep)
    edges_df_filtered = filter_edges_df(edges_df, indexes_to_keep, new_index_mapping)
    node_features_dict_filtered = filter_node_features(node_features_dict, new_index_mapping, indexes_to_keep)
    return target_df_filtered, edges_df_filtered, node_features_dict_filtered


def create_binary_features_tensor(node_feats):
    print('\nAdd features and targets to the graph...')

    # Determine the number of unique integer features in the dataset
    unique_features = set(feature for features_list in node_feats.values() for feature in features_list)
    num_unique_features = len(unique_features)
    print('number of unique features:', len(unique_features))

    # Create an empty tensor to hold the binary node features
    num_nodes = len(node_feats)
    binary_features_tensor = torch.zeros(num_nodes, num_unique_features, dtype=torch.float32)

    # Iterate through the node indices and features in the node_features dictionary
    for node_idx, features_list in node_feats.items():
        # Iterate through the integer features and set the corresponding cell to 1
        for feature in features_list:
            feature_index = list(unique_features).index(feature)
            binary_features_tensor[int(node_idx)][feature_index] = 1

    print('Done.')
    return binary_features_tensor


def create_masks(num_nodes, t_ratio):
    # Define the ratio of nodes for training (it can be adjusted as needed)
    train_ratio = t_ratio  # 60% for training, 20% for validation, 20% for testing

    # Set the random seed for reproducibility (optional)
    np.random.seed(0)
    # Generate random indices for shuffling the nodes
    rand_indices = np.random.permutation(num_nodes)

    # Split the nodes based on the defined ratios
    train_idx = rand_indices[:int(train_ratio * num_nodes)]
    val_idx = rand_indices[int(train_ratio * num_nodes):int((train_ratio + 0.2) * num_nodes)]
    test_idx = rand_indices[int((train_ratio + 0.2) * num_nodes):]

    # Create train, validation, and test masks
    train_mask = torch.tensor(np.zeros(num_nodes, dtype=bool))
    val_mask = torch.tensor(np.zeros(num_nodes, dtype=bool))
    test_mask = torch.tensor(np.zeros(num_nodes, dtype=bool))

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def plot_graph(graph):
    nx_graph = graph.to_networkx()
    # Define the layout
    pos = nx.spring_layout(nx_graph)

    # Save the plot as an image (e.g., PNG)
    plt.figure(figsize=(8, 8))
    nx.draw(nx_graph, pos, with_labels=True, node_color='skyblue', node_size=50, font_size=10, font_color='black')
    plt.savefig('graph_noloop_plot.png')  # Specify the filename and file format


def normalize_target_values(target_df):
    # print(min(target_values), max(target_values))
    df_min_max_scaled = target_df.copy()
    min_val, max_val = df_min_max_scaled['target'].min(), df_min_max_scaled['target'].max()
    df_min_max_scaled['target'] = (df_min_max_scaled['target'] - min_val) / (max_val - min_val)
    # print(df_min_max_scaled['target'])
    return df_min_max_scaled


def boxplot_targets(target_values):
    unique_targets = set(target_values)
    num_unique_targets = len(unique_targets)
    print(num_unique_targets)

    # Create a box plot
    plt.figure(figsize=(8, 6))
    plot = plt.boxplot(target_values, vert=False, whis=1.5, showfliers=False)  # vert=False for horizontal box plot
    plt.title('Box Plot of Target Values')
    plt.xlabel('Target Values')
    plt.show()


def create_graph(edges_df, binary_node_features, target_df):
    # Extract source and destination nodes from edges
    src_nodes = edges_df['id1'].tolist()
    dst_nodes = edges_df['id2'].tolist()

    # Create a DGL graph
    graph = dgl.graph((src_nodes, dst_nodes), num_nodes=len(binary_node_features))
    graph = dgl.add_self_loop(graph)  # Optionally add self-loops if needed

    # Save a plot of graph
    # plot_graph(graph)

    # Add the binary feature vectors to the DGL graph's node data
    graph.ndata['feature'] = binary_node_features
    # print([graph.ndata['feature'][2098][2111]])

    # Add target values to the graph
    target_values = target_df['target'].tolist()
    graph.ndata['target'] = torch.tensor(target_values, dtype=torch.float32)  # dtype could be int64?

    # Boxplot the target values and their distribution
    # boxplot_targets(target_values)

    # Create and Add the masks to the DGL graph's node data
    num_nodes = graph.number_of_nodes()
    train_mask, val_mask, test_mask = create_masks(num_nodes, t_ratio=0.6)
    graph.ndata['train_mask'], graph.ndata['val_mask'], graph.ndata['test_mask'] = train_mask, val_mask, test_mask

    return graph


def show_graph_details(graph, graph_details):
    features, targets, train_mask, val_mask, test_mask = graph_details
    print('\nGraph:', graph)
    print('\nfeatures:')
    print(features)
    print('\ntargets:')
    print(targets)
    print('\nTrain, Validation, and Test Masks:')
    print(train_mask)
    print(val_mask)
    print(test_mask)


def RMSE_loss(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))


def rsquared_loss(pred, labels):
    tot = ((labels - labels.mean()) ** 2).sum()
    res = ((labels - pred) ** 2).sum()
    r2 = 1 - res / tot
    return r2


def run_model(gnn, graph, graph_details, hidden_dim, num_heads=8, output_dim=1):
    features, targets, train_mask, val_mask, test_mask = graph_details
    print('\n* Model: ' + gnn + ' *')

    # Initialize the GAT model
    in_dim = features.shape[1]
    # hidden_dim = 8
    # hidden_dim = 10  # 32
    # num_heads = 8
    # num_heads = 10
    # output_dim = 1

    if gnn == 'GAT':
        model = GAT(in_dim, hidden_dim, output_dim, num_heads)
        model = GATNodeRegression(in_dim, hidden_dim, num_heads, output_dim=len(targets))
    elif gnn == 'GATv2':
        model = GATv2(in_dim, hidden_dim, output_dim, num_heads)
        model = GATv2NodeRegression(in_dim, hidden_dim, num_heads, output_dim=len(targets))
    elif gnn == 'GCN':
        model = GCN(in_dim, hidden_dim, output_dim=len(targets))
        model = GCNNodeRegression(in_dim, hidden_dim, output_dim=len(targets))
    elif gnn == 'SAGE':
        model = SAGE(in_dim, hidden_dim, out_feats=len(targets))
        # model = Model(in_dim, hidden_dim, len(targets))
        model = SAGENodeRegression(in_dim, hidden_dim, (targets))

    # Set up the training loop

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression
    # criterion = MeanAbsolutePercentageError()  # Mean Squared Error loss for regression
    # criterion = MeanAbsoluteError()  # Mean Squared Error loss for regression
    # criterion = RMSE_loss  # Mean Squared Error loss for regression

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    optimizer = optim.Adam(model.parameters(), lr=0.007)
    # Use SGD optimizer
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    num_epochs = 200
    num_epochs = 500
    num_epochs = 100

    test_loss = []

    # Training loop
    for epoch in range(num_epochs):
        t0 = time.time()

        model.train()
        optimizer.zero_grad()
        # logits = model(graph, features)
        # loss = criterion(logits[train_mask], targets[train_mask])
        predictions = model(graph, features).squeeze()  # Squeeze to remove the extra dimension
        loss = criterion(predictions[train_mask], targets[train_mask])  # Use MSE loss for regression
        # print('\npredictions:', predictions[train_mask])
        # print('targets:', targets[train_mask])
        # print(loss, targets[train_mask]-predictions[train_mask])

        # loss.backward()
        # optimizer.step()

        model.eval()
        with torch.no_grad():
            # logits = model(graph, features)
            # predictions = logits.argmax(dim=1)
            # train_acc = (predictions[train_mask] == targets[train_mask]).float().mean().item()
            # val_acc = (predictions[val_mask] == targets[val_mask]).float().mean().item()
            # test_acc = (predictions[test_mask] == targets[test_mask]).float().mean().item()
            predictions = model(graph, features).squeeze()
            mse = nn.MSELoss()
            # mse = MeanAbsolutePercentageError()
            # mse = MeanAbsoluteError()
            # mse = RMSE_loss

            train_mse = (mse(predictions[train_mask], targets[train_mask]).item())  # * 10 ** -10
            val_mse = mse(predictions[val_mask], targets[val_mask]).item()  # * 10 ** -10
            test_mse = mse(predictions[test_mask], targets[test_mask]).item()  # * 10 ** -10
            test_loss.append(test_mse)

            # torch.set_printoptions(threshold=6)
            # print('test predictions: ', predictions[test_mask])
            # print('test targets:     ', targets[test_mask])
            # print()
            # torch.set_printoptions(threshold=None)

        model.train()
        loss.backward()
        optimizer.step()

        model.eval()

        # print('predictions:', predictions[train_mask])
        # print('targets:', targets[train_mask])
        # print()

        duration = time.time() - t0
        if epoch == num_epochs - 1:
            print(
                f'Epoch: {epoch + 1}, Loss: {loss.item():.4f}, Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}, Test MSE: {test_mse:.4f}, Time: {duration:.4f}')
    print(min(test_loss))
    # if loss.item() < 0.00005:
    #     break

    torch.set_printoptions(threshold=6)
    print('\ntrain predictions:', predictions[train_mask])
    print('train targets:    ', targets[train_mask])
    print('\ntest predictions: ', predictions[test_mask])
    print('test targets:     ', targets[test_mask])
    torch.set_printoptions(threshold=None)


def get_graph_details(graph):
    features = graph.ndata['feature']
    targets = graph.ndata['target']
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    graph_details = [features, targets, train_mask, val_mask, test_mask]
    return graph_details


def main():
    # Determine the dataset to implement
    dataset_name = 'chameleon'
    print(f'* GNN on \"wiki-{dataset_name}\" dataset *')

    # Load wiki dataset
    edges_df, node_features, target_df = read_files(dataset_files['dataset_files_' + dataset_name])

    # Preprocess & Remove outliers
    target_df, edges_df, node_features = remove_outliers(target_df, edges_df, node_features)

    # Normalize target values
    target_df = normalize_target_values(target_df)

    # Convert dictionary node features to tensor binary vectors
    binary_node_features = create_binary_features_tensor(node_features)

    # Create graph from data files
    graph = create_graph(edges_df, binary_node_features, target_df)

    # Define and show graph parameters
    graph_details = get_graph_details(graph)
    show_graph_details(graph, graph_details)

    # run GATv1 on the graph
    run_model(gnn='GAT', graph=graph, graph_details=graph_details, hidden_dim=8, num_heads=8)

    # run GATv2 on the graph
    run_model(gnn='GATv2', graph=graph, graph_details=graph_details, hidden_dim=8, num_heads=8)

    # run GCN on the graph
    run_model(gnn='GCN', graph=graph, graph_details=graph_details, hidden_dim=16)

    # run SAGE on the graph
    run_model(gnn='SAGE', graph=graph, graph_details=graph_details, hidden_dim=16)


if __name__ == '__main__':
    main()
