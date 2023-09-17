import matplotlib.pyplot as plt
import numpy as np

def detect_anomalies():
    # Generate some synthetic data for demonstration
    data_x = np.random.rand(100)
    data_y = np.random.rand(100)
    
    # Let's assume points where x + y > 1.5 are anomalies
    anomalies_x = [x for x, y in zip(data_x, data_y) if x + y > 1.5]
    anomalies_y = [y for x, y in zip(data_x, data_y) if x + y > 1.5]
    
    # Plotting the data
    plt.scatter(data_x, data_y, color='blue', label='Normal Data')
    plt.scatter(anomalies_x, anomalies_y, color='red', label='Anomalies')
    
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Anomaly Detection')
    plt.legend()
    
    plt.show()

# Uncomment the line below to test the function
# detect_anomalies()
