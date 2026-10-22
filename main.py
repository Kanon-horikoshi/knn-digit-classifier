from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
# Load handwritten digit dataset
digits = load_digits()

# Images and labels
X = digits.data
y = digits.target

print("Data shape:", X.shape)
print("Labels shape:", y.shape)

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"Label: {y[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()


# L1 distance
def l1_distance(a, b):
    total = 0

    for x, y in zip(a, b):
        total = total + abs(x - y) 

    return total


a = [1, 2]
b = [4, 6]

# 1 x=1y=4,2  x=2y=6
print("L1 distance:", l1_distance(a, b))

# L2 distance
def l2_distance(a, b):
    total = 0

    for x, y in zip(a, b):
        total = total + (x-y)**2

    return (total)**0.5

print("L2 distance:", l2_distance(a, b))





from sklearn.model_selection import train_test_split

# Separate data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Calculate the distance to every training image
distances = []

for image in X_train:
    distance = l2_distance(X_test[0], image)
    distances.append(distance)

# Find the closest image
closest_index = distances.index(min(distances))

# Predict the digit
prediction = y_train[closest_index]

print("AI prediction:", prediction)
print("Correct answer:", y_test[0])




from collections import Counter

# Sort by distance (smallest first)
neighbors = sorted(
    zip(distances, y_train),
    # 距離と正解ラベルをペア
    key=lambda pair: pair[0]
)

for k in [1, 3, 5]:
    nearest_labels = [
        label for distance, label in neighbors[:k]
    ]
    # 画像を距離が近い順に並べる 距離が近いListの最初の3個だけ取り出す

    prediction = Counter(nearest_labels).most_common(1)[0][0]
    #　それぞれの数字が何回出てきたかを数える

    print(f"K={k}, Prediction={prediction}")

print("Correct answer:", y_test[0])





# Test KNN accuracy for different K values

accuracies = []

for k in [1, 3, 5]:
    correct = 0
    total = 50
    #50枚をテストする

    for i in range(total):
        distances = []

        for image, label in zip(X_train, y_train):
            d = l2_distance(X_test[i], image)
            distances.append((d, label))

        distances.sort(key=lambda pair: pair[0])

        nearest_labels = [
            label for distance, label in distances[:k]
        ]

        prediction = Counter(nearest_labels).most_common(1)[0][0]

        if prediction == y_test[i]:
            correct += 1
        elif k == 5:
            print("AI predicted:", prediction)
            print("Actual answer:", y_test[i])

            # Error Analysis
            print("3 nearest labels:")
            print([label for distance, label in distances[:3]])

            print("5 nearest labels:")
            print([label for distance, label in distances[:5]])

            print("5 closest images:")

            for rank, (distance, label) in enumerate(distances[:5], start=1):
                print(f"{rank}: Label={label}, Distance={distance:.2f}")


            plt.figure(figsize=(4, 4))
            plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
            plt.title(f"Predicted: {prediction}, Actual: {y_test[i]}")
            plt.axis('off')
            plt.show()

    accuracy = correct / total
    accuracies.append(accuracy)

    print(f"K={k}, Accuracy={accuracy:.2%}")
    #パーセントで小数第2位まで

#グラフ
k_values = [1, 3, 5]
print("K values:", k_values)
print("Accuracies:", accuracies)

plt.figure(figsize=(7, 4))  # 新しいグラフを作る

plt.plot(k_values, accuracies, marker='o')
plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy Comparison")
plt.xticks(k_values)
plt.ylim(0, 1.05)
plt.grid(True)
plt.show()
