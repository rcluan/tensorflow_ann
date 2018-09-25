
TRAINING_PERCENTAGE = 0.8

class DataProcessor:

  def __init__(self, data):

    self.current_pointer = 0
    self.raw_data = data

    processed_data = {
      'input_labels': [],
      'all_rows': [],
      'training_rows': [],
      'testing_rows': []
    }
    
    for key, value in enumerate(self.raw_data):
      if(key > 0):
        processed_data['input_labels'].append(value)

    
    for key, row in self.raw_data.iterrows():
      row_content = []
      for label in processed_data['input_labels']:
        row_content.append(row[label])
      
      processed_data['all_rows'].append(row_content)

    total_rows = len(processed_data['all_rows'])

    training_length = int(total_rows * TRAINING_PERCENTAGE)

    processed_data['training_rows'] = processed_data['all_rows'][:training_length]
    processed_data['testing_rows'] = processed_data['all_rows'][training_length:]

    self.processed_data = processed_data

  def next_batch(self, batch_size):
    batch = self.processed_data[self.current_pointer:batch_size]
    self.current_pointer = self.current_pointer + batch_size
    return batch
