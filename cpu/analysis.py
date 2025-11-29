import matplotlib.pyplot as plt
import pandas as pd
import io

# Load the provided data

def plot_avg_accuracy(csv_files, file_name=""):
    """Plots the average train and test accuracy per round for multiple CSV files."""
    plt.figure(figsize=(12, 6))
    colors = ['blue', 'orange', 'green', 'red', 'purple'] # Add more colors if needed
    
    for i, csv_file in enumerate(csv_files):
        if isinstance(csv_file, str): # Check if it is a filename path or direct content
           df = pd.read_csv(csv_file)
           label_prefix = csv_file
        else: # Expecting a file object
            df = pd.read_csv(csv_file)
            label_prefix = f"File {i+1}"

        avg_acc = df.groupby('Round')[['Train_Acc', 'Test_Acc']].mean()
        
        plt.plot(avg_acc.index, avg_acc['Train_Acc'], linestyle='--', marker='o', color=colors[i % len(colors)], label=f'{label_prefix} Train Avg')
        plt.plot(avg_acc.index, avg_acc['Test_Acc'], linestyle='-', marker='x', color=colors[i % len(colors)], label=f'{label_prefix} Test Avg')

    plt.title('Average Train and Test Accuracy Comparison')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    if file_name:
        plt.savefig(file_name, dpi=300)
    plt.show()

def plot_client_accuracy(csv_file):
    """Plots the train and test accuracy for each client in a single CSV file."""
    if isinstance(csv_file, str) and "\n" in csv_file:
        df = pd.read_csv(io.StringIO(csv_file))
    else:
        df = pd.read_csv(csv_file)

    unique_clients = df['Client_ID'].unique()

    colors = plt.get_cmap(None, len(unique_clients))
    
    # Plot Training Accuracy
    plt.figure(figsize=(12, 6))
    for client_id in unique_clients:
        color = colors(client_id % 10)
        client_data = df[df['Client_ID'] == client_id]
        plt.plot(client_data['Round'], client_data['Train_Acc'], marker='o', label=f'Client {client_id} Train', color=color)
        plt.plot(client_data['Round'], client_data['Test_Acc'], marker='x', label=f'Client {client_id} Test', linestyle='--', color=color)
    
    plt.title('Training and Test Accuracy per Client')
    plt.xlabel('Round')
    plt.ylabel('Training Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('client_accuracy.pdf', dpi=300)
    plt.show()


# Example Usage:
# Passing the string data directly for demonstration
def main():
    csv_sparse = "results_sparse_20_rounds.csv"
    csv_dense = "results_dense_20_rounds.csv"
    csv_2_sparse = "results_sparse_2_clients_16_rounds.csv"
    csv_4_sparse = "results_sparse_4_clients_16_rounds.csv"
    csv_8_sparse = "results_sparse_8_clients_16_rounds.csv"

    csv_topk = "results_sparse_topk.csv"
    csv_random = "results_sparse_random.csv"
    csv_stochastic = "results_sparse_stochastic.csv"

    csv_none_sparse = "results_sparse_none_16_rounds.csv"
    csv_none_dense = "results_dense_none_16_rounds.csv"


    plot_avg_accuracy([csv_none_sparse, csv_none_dense], file_name="treste.pdf") 
    # plot_avg_accuracy([csv_sparse, csv_dense], file_name="avg_accuracy_sparse_vs_dense.pdf") 
    # plot_client_accuracy(csv_sparse)
    # plot_avg_accuracy([csv_2_sparse, csv_4_sparse, csv_8_sparse], file_name="avg_accuracy_varying_clients.pdf")
    # plot_avg_accuracy([csv_topk, csv_random, csv_stochastic], file_name="avg_accuracy_sparsity_methods.pdf")

if __name__ == "__main__":
    main()

