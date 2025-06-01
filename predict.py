from model_predict import predict

# Test
if __name__ == '__main__':
    test_image = 'pampers.webp'  # Replace with your image path
    pred = predict(test_image)
    print(f"🧠 Predicted class: {pred}")