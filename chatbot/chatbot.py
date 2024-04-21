
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd

# Load CSV data
data = pd.read_csv('D:/Hack/Hack/chatbot/final3.csv', encoding='latin1')

# Define the bag_of_words function
def bag_of_words(sentence, words):
    # Tokenize the sentence
    sentence_words = word_tokenize(sentence)
    # Stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # Initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

# Tokenize patterns
# nltk.download('punkt')
all_words = []
tags = []
patterns = []

for index, row in data.iterrows():
    pattern_sentence = row['Description']
    color = row['Color']
    clothing_type = row['type']
    all_words.extend(word_tokenize(pattern_sentence))
    tags.extend([color, clothing_type])
    patterns.append((pattern_sentence, color, clothing_type))

# Preprocess words
stemmer = PorterStemmer()
ignore_words = ['?', '.', '!']
all_words = [stemmer.stem(w.lower()) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Define X_train (input) and Y_train (output) data
X_train = []
Y_train_color = []
Y_train_type = []
for (pattern_sentence, color, clothing_type) in patterns:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    Y_train_color.append(tags.index(color))
    Y_train_type.append(tags.index(clothing_type))

X_train = np.array(X_train)
Y_train_color = np.array(Y_train_color)
Y_train_type = np.array(Y_train_type)

# Define the PyTorch model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

# Define training parameters
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
# learning_rate = 0.001
# num_epochs = 1000

# # Initialize the model, loss function, and optimizer
# model = NeuralNet(input_size, hidden_size, output_size)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # Train the model
# for epoch in range(num_epochs):
#     inputs = torch.from_numpy(X_train).float()
#     targets_color = torch.from_numpy(Y_train_color).long()
#     targets_type = torch.from_numpy(Y_train_type).long()

#     # Forward pass for color prediction
#     optimizer.zero_grad()
#     outputs_color = model(inputs)
#     loss_color = criterion(outputs_color, targets_color)
#     loss_color.backward()
#     optimizer.step()

#     # Forward pass for type prediction
#     optimizer.zero_grad()
#     outputs_type = model(inputs)
#     loss_type = criterion(outputs_type, targets_type)
#     loss_type.backward()
#     optimizer.step()

#     if (epoch+1) % 100 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss (Color): {loss_color.item():.4f}, Loss (Type): {loss_type.item():.4f}')

# # Save the trained model
# torch.save(model.state_dict(), 'chatbot_model.pth')

# Load the trained model
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('chatbot_model.pth'))
model.eval()
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(filtered_tokens)

def preprocess_column(column):
    return [preprocess_text(text) for text in column]

lemmatizer = WordNetLemmatizer()
def process_user_input(user_input):
   import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd

# Download NLTK resources


# Sample training data (replace this with your actual dataset)


# Specify the columns you want to preprocess
training_data = pd.read_csv('D:/Hack/Hack/chatbot/final3.csv', encoding='latin1')

# Preprocess training data
columns_to_preprocess = ['Color', 'type']

# Initialize dictionaries to store preprocessed X_train lists for each target variable
X_train_dict = {}

# Preprocess each column and create X_train lists for each target variable
for column_name in columns_to_preprocess:
    X_train_dict[column_name] = preprocess_column(training_data[column_name])
y_train_color = training_data['Color']
y_train_description = training_data['type']

# Create TF-IDF vectorizer for Color
tfidf_vectorizer_color = TfidfVectorizer()
X_train_tfidf_color = tfidf_vectorizer_color.fit_transform(X_train_dict['Color'])

# Create TF-IDF vectorizer for Type
tfidf_vectorizer_type = TfidfVectorizer()
X_train_tfidf_type = tfidf_vectorizer_type.fit_transform(X_train_dict['type'])

# Split the data into training and testing sets for each target variable
X_train_color, X_test_color, y_train_color, y_test_color = train_test_split(X_train_tfidf_color, y_train_color, test_size=0.5, random_state=42)
X_train_description, X_test_description, y_train_description, y_test_description = train_test_split(X_train_tfidf_type, y_train_description, test_size=0.5, random_state=42)

# Train SVM model for each target variable
svm_model_color = SVC(kernel='linear')
svm_model_color.fit(X_train_color, y_train_color)

svm_model_description = SVC(kernel='linear')
svm_model_description.fit(X_train_description, y_train_description)
def process_user_input(user_input):

    # Sample test query
    test_query = user_input

    # Preprocess test query
    X_test = [preprocess_text(test_query)]

    # Transform test query using TF-IDF vectorizer for each target variable
    X_test_tfidf_color = tfidf_vectorizer_color.transform(X_test)
    X_test_tfidf_type = tfidf_vectorizer_type.transform(X_test)

    # Predict for each target variable
    predicted_color = svm_model_color.predict(X_test_tfidf_color)
    predicted_description = svm_model_description.predict(X_test_tfidf_type)

    # Output predicted values
    print("Predicted Color:", predicted_color[0])
    print("Predicted Type:", predicted_description[0])

    # Retrieve URL and image for the predicted color (assuming the dataset contains this information)
    predicted_products = training_data[(training_data['Color'] == predicted_color[0]) & (training_data['type'] == predicted_description[0])].head(5)

    # Display product information
    response = []
    for index, product in predicted_products.iterrows():
        response.append({
            'image': product['URL_image'],
            'url': product['Product_URL']
        })
    
    return response