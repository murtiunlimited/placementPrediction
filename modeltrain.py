import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("./dataset/preprocessed_placement_data.csv")

X = df.drop('status', axis=1)
y = df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
pickle.dump(model, open("./app/placement_model.pkl", 'wb'))

# Save LabelEncoder for categorical features
le = LabelEncoder()
cat_cols = ['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation','status']
for col in cat_cols:
    le.fit(df[col])
pickle.dump(le, open("./app/labelencoder.pkl", 'wb'))

# Save Scaler
scaler = StandardScaler()
num_cols = ['ssc_p','hsc_p','degree_p','etest_p','mba_p']
scaler.fit(df[num_cols])
pickle.dump(scaler, open("./app/scaler.pkl", 'wb'))