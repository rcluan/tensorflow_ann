import numpy as np

TRAINING_PERCENTAGE = 0.7392

class DataProcessor:

  def __init__(self, data):

    self.current_pointer = 0
    self.raw_data = data

    processed_data = {
      'input_labels': [],
      'all_rows': [],
      'all_labels': [],
      'x_train_rows': [],
      'y_train_labels': [],
      'x_validate_rows': [],
      'y_validate_labels': []
    }
    
    for key, value in enumerate(self.raw_data):
      if(key > 0):
        processed_data['input_labels'].append(value)

    
    for key, row in self.raw_data.iterrows():
      row_content = []
      label_content = []
      for label in processed_data['input_labels']:
        row_content.append(row[label])
        if label == 'v_anemo2':
          label_content.append(row[label])
      
      processed_data['all_rows'].append(row_content)
      processed_data['all_labels'].append(label_content)

    total_rows = len(processed_data['all_rows'])

    training_length = int(total_rows * TRAINING_PERCENTAGE)

    processed_data['x_train_rows'] = np.array(processed_data['all_rows'][:training_length])
    processed_data['y_train_labels'] = np.array(processed_data['all_labels'][:training_length])
    
    processed_data['x_validate_rows'] = np.array(processed_data['all_rows'][training_length:])
    processed_data['y_validate_labels'] = np.array(processed_data['all_labels'][training_length:])

    self.processed_data = processed_data