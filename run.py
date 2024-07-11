from Core.data import Data
from Core.config import Column
from Core.model import LSTM_Trainer

# Load and preprocess the data
data = Data()
data.read('Data/T.csv')
data.check_null_values()
data.clean_data()
print(Column.OPEN.value)
data.print_head()
data.print_description()
data.normalize()
data.visualize(Column.OPEN.value)
data.visualize(Column.CLOSE.value)

# Train the LSTM model and make predictions
trainer = LSTM_Trainer(data.dataframe, data.scaler)
trainer.build_and_train_lstm()

# Save the trained model
model_path = 'saved_model/lstm_model.h5'
trainer.model.save(model_path)
print(f'Model saved to {model_path}')

# To load the model, uncomment the following lines:
# from tensorflow.keras.models import load_model
# trainer.model = load_model(model_path)
# print(f'Model loaded from {model_path}')

trainer.predict_and_plot()
trainer.evaluate_model()
