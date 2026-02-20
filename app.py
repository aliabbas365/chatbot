# Import necessary libraries
try:
	import pandas as pd
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import StandardScaler
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import accuracy_score, classification_report
	import pickle 
except Exception as e:
	print("Error importing libraries:", e)
	raise


def main():
	# Load the dataset
	try:
		data = pd.read_csv('DATA.csv')
	except FileNotFoundError:
		print("ERROR: 'DATA.csv' not found in the current working directory.")
		print("Make sure you're running the script from c:\\Users\\786\\chatbot or provide the correct path to DATA.csv.")
		return

	# Handle missing values (if any) - only compute means for numeric columns to avoid TypeError
	data.fillna(data.mean(numeric_only=True), inplace=True)

	# Encode categorical variables (one-hot encoding)
	data = pd.get_dummies(data, drop_first=True)

	# Quick sanity check: show columns and shape
	print(f"Data loaded: shape={data.shape}")
	print(f"Columns (first 10): {list(data.columns[:10])}")

	# Feature selection (use all columns except the target 'Attrition' column)
	if 'Attrition_Yes' not in data.columns:
		print("ERROR: expected target column 'Attrition_Yes' not found after encoding. Available columns:")
		print(list(data.columns))
		return

	X = data.drop(columns=['Attrition_Yes'])  # 'Attrition_Yes' is the target column after encoding
	y = data['Attrition_Yes']  # Target variable

	# Save the training feature columns so the server can reindex incoming requests
	feature_columns = list(X.columns)
	with open('feature_columns.pkl', 'wb') as f:
		pickle.dump(feature_columns, f)

	# Split data into training and testing sets (80% train, 20% test)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Scale numerical features
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	# Minimal model fit to confirm everything runs
	clf = RandomForestClassifier(random_state=42)
	clf.fit(X_train_scaled, y_train)
	peds = clf.predict(X_test_scaled)
	print("Accuracy:", accuracy_score(y_test, peds))

	# Detailed classification metrics
	print("\nClassification report:")
	print(classification_report(y_test, peds))

	# Save the trained model and scaler for later use
	with open('rf_model.pkl', 'wb') as f:
		pickle.dump(clf, f)
	with open('scaler.pkl', 'wb') as f:
		pickle.dump(scaler, f)
	print("Saved trained model to rf_model.pkl and scaler to scaler.pkl")


if __name__ == "__main__":
	main()







