import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Sinir Ağı Yapısı
class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_size, learnin_rate=0.1):
        # Ağırlıklar
        self.weights1 = np.random.rand(input_size, hidden_size) - 0.5
        self.weights2 = np.random.rand(hidden_size, output_size) - 0.5

        # Biaslar
        self.bias1 = np.random.rand(hidden_size) - 0.5
        self.bias2 = np.random.rand(output_size) - 0.5

        # Öğrenme Katsayısı
        self.learning_rate = learnin_rate

        # Başlangıçta atanan ağırlık ve biasları sakla
        self.initial_weights1 = self.weights1.copy()
        self.initial_weights2 = self.weights2.copy()
        self.initial_bias1 = self.bias1.copy()
        self.initial_bias2 = self.bias2.copy()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # İleri Yayılım
    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        output = self.sigmoid(self.z2)
        return output

    # Geri Yayılım
    def backward(self, X, y, output):
        # Çıkış hatası ve gradyan hesaplaması
        error_output = y - output
        d_output = error_output * self.sigmoid_derivative(output)

        # Gizli katman hatası ve gradyan hesaplaması
        error_hidden = d_output.dot(self.weights2.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.a1)

        # Ağırlık ve bias güncelleme
        self.weights2 += self.a1.T.dot(d_output) * self.learning_rate
        self.weights1 += X.T.dot(d_hidden) * self.learning_rate
        self.bias2 += np.sum(d_output, axis=0) * self.learning_rate
        self.bias1 += np.sum(d_hidden, axis=0) * self.learning_rate

    # Eğitim fonksiyonu
    def train(self, X, y, epochs):
        total_loss = 0  # Toplam hatayı saklamak için değişken
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

            # Her epoch'ta hatayı hesapla ve toplam hata değerine ekle
            loss = np.mean(np.square(y - output))
            total_loss += loss

        # Eğitim sonunda toplam hatayı yazdır
        print(f"Eğitim Sonu - Toplam Hata (Loss): {total_loss:.6f}")

        # Eğitim öncesi ve sonrası ağı görselleştir
        self.visualize_network(X, initial=True)  # Eğitim öncesi
        self.visualize_network(X, initial=False)  # Eğitim sonrası

    # Sinir ağının görselleştirilmesi
    def visualize_network(self, X, initial=True):
        G = nx.DiGraph()

        # Giriş nöronları
        input_nodes = [f"Input {i+1}" for i in range(X.shape[1])]
        G.add_nodes_from(input_nodes)

        # Gizli nöronlar
        hidden_nodes = [f"Hidden1_{i+1}" for i in range(self.weights1.shape[1])]
        G.add_nodes_from(hidden_nodes)

        # Çıkış nöronları
        output_nodes = [f"Output {i+1}" for i in range(self.weights2.shape[1])]
        G.add_nodes_from(output_nodes)

        # Ağırlıkları ve biasları belirle
        if initial:
            weights1 = self.initial_weights1
            weights2 = self.initial_weights2
            bias1 = self.initial_bias1
            bias2 = self.initial_bias2
            title = "Eğitim Öncesi Ağı"
        else:
            weights1 = self.weights1
            weights2 = self.weights2
            bias1 = self.bias1
            bias2 = self.bias2
            title = "Eğitim Sonrası Ağı"

        # Girişten gizliye bağlantılar ve ağırlıklar
        edge_labels = {}
        for i, input_node in enumerate(input_nodes):
            for j, hidden_node in enumerate(hidden_nodes):
                G.add_edge(input_node, hidden_node)
                edge_labels[(input_node, hidden_node)] = f'{weights1[i,j]:.2f}'

        # Gizliden çıkışa bağlantılar ve ağırlıklar
        for i, hidden_node in enumerate(hidden_nodes):
            for j, output_node in enumerate(output_nodes):
                G.add_edge(hidden_node, output_node)
                edge_labels[(hidden_node, output_node)] = f'{weights2[i,j]:.2f}'

        # Katmanların konumunu ayarlama
        pos = {}
        layer_width = 3  # Katmanlar arası genişlik
        layer_height = 2  # Katmandaki nöronlar arası yükseklik

        # Giriş katmanının konumu
        for i, node in enumerate(input_nodes):
            pos[node] = (0, i * layer_height)

        # Gizli katmanın konumu
        for i, node in enumerate(hidden_nodes):
            pos[node] = (layer_width, i * layer_height)

        # Çıkış katmanının konumu
        for i, node in enumerate(output_nodes):
            pos[node] = (2 * layer_width, i * layer_height)

        # Ağı çizme
        plt.figure(figsize=(12, 8))
        
        # Düğümleri çizme
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=3000)
        
        # Font ayarları
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        nx.draw_networkx_edges(G, pos)
        
        # Ağırlıkları çizme
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, label_pos=0.75)  # Ağırlıkları bağlantılara daha yakın yerleştir

        # Bias değerlerini ekleme
        for i, node in enumerate(hidden_nodes):
            plt.annotate(f'bias: {bias1[i]:.2f}',
                        xy=pos[node],
                        xytext=(10, -10),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        fontsize=8)
        
        for i, node in enumerate(output_nodes):
            plt.annotate(f'bias: {bias2[i]:.2f}',
                        xy=pos[node],
                        xytext=(10, -10),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        fontsize=8)

        plt.title(title)
        plt.axis('off')
        plt.show()

# Kullanıcıdan gizli katman nöron sayısını alma
hidden_size = int(input("Gizli katman nöron sayısını girin: "))

# Veri seti ve eğitim
X = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 1], [0, 0, 1, 1]])
y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Giriş ve hedef değerlerini yazdırma
print("Giriş Değerleri:")
print(X)
print("Hedef Değerleri:")
print(y)

# Sinir ağını oluşturma (4 giriş, kullanıcıdan alınan gizli nöron sayısı, 2 çıkış)
nn = NeuralNetwork(input_size=4, hidden_size=hidden_size, output_size=2)

# Eğitim 
nn.train(X, y, epochs=10000)
