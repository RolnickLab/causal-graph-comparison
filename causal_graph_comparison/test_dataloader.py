import torch
import numpy as np

class SineTestDataset(torch.utils.data.Dataset):

	"""A simple dataset that generates sine wave data for testing models.

	Args:
		num_samples (int): Number of samples to generate
		tau (int, optional): Number of past timesteps to use. Defaults to 30.
		future (int, optional): Number of future timesteps to predict. Defaults to 1.
		test (bool, optional): Whether this is test data. If True, adds offset. Defaults to False.

	Returns:
		tuple: (past_data, future_data) where:
			- past_data has shape (tau, 1) containing tau timesteps of history
			- future_data has shape (future, 1) containing future timesteps to predict
	"""

	def __init__(self, num_samples, tau=30, future=1, test=False):
		x = np.linspace(0,100, num_samples) # from, to, number of samples… will give [0, 0.1, 0.2, …]

		offset = 0
		if test:
			offset = 25 # arbitrary number to be added to x

		y = np.sin(x + offset) # this will be our data
		self.data = torch.from_numpy(y)
		self.tau = tau
		self.future = future


	def __len__(self):
		return len(self.data) - self.tau - self.future - 1


	def __getitem__(self, idx):
		current_timestep = idx + self.tau
		past_data = self.data[current_timestep-self.tau : current_timestep]
		future_data = self.data[current_timestep : current_timestep + self.future]
		
		# Ensure past_data has shape (tau,) and future_data has shape (future,)
		past_data = past_data.float().unsqueeze(1)
		future_data = future_data.float().unsqueeze(1)
		
		# For LSTM, we need past_data to be (tau,) and future_data to be (future,)
		# But the model expects (batch, seq_len, features), so we'll reshape in the model
		return past_data, future_data

if __name__ == "__main__":
    dataset_train = SineTestDataset(num_samples=1000)
    dataset_test = SineTestDataset(num_samples=1000, test=True)
    
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=100, shuffle=False)

    for i, (past_data, future_data) in enumerate(train_loader):
        print(past_data.shape)
        print(future_data.shape)
        break