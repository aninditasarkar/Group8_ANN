# Classification of Virtual Machine Workloads using Artifical Neural Network(ANN)

<p align="left">
<img src="ANN_layers.png?raw=true"
  alt=""width="400" height="250">
</p>
In this project we attempt to classify the virtual machines on the basis of CPU and Memory utilization (over utilization and under utilization). Virtual machines having CPU and memory utilization above or equal to 90% will be classified as overutilized and the ones with utilization below 20% will be classified as underutilized. We used a neural network trained through supervised learning by using real VM workload traces.
Moreover, we will use popular Python libraries such as numpy , os . We will use sklearn for the classification.

## Little background in ANN

Neural networks adapt themselves to the changing input so that the network generates the best possible result without the need to redesign the output criteria. The functionality of neural networks is often compared to the one of the multiple linear regression, where one uses multiple input features, also called independent variables, to predict the output variable, the dependent variable. In case of Neural Network we also use input features, referred as **Input Layer Neurons** to get information and learn about the outcome variable, referred as **Output Layer**.The main difference between such regression and Neural Network is that, in the case the former the process runs in one iteration by minimizing the sum of the squared residuals (similar to cost function), whereas in case of Neural Network there is an intermediate step portrayed by the **Hidden Layer Neurons** which are used to get signals from the input layers and learn about the observations over and over again until the goal is achieved, the cost is minimized and no improvement is possible. So, one can say that ANNs are much more sophisticated than multiple linear regression.

# Description of Data

To train our neural networks we used the Datacenter_dataset from the GitHub account- https://github.com/sevaiq . These traces were stored in .csv files with the schema shown in the Table 1 below.

![](pictures/table1.png)

<p align="left">
<img src="table1.png?raw=true"
  alt=""width="500" height="250">
</p>

![](pictures/formula.jpeg)

<p align="left">
<img src="formula.jpeg?raw=true"
  alt=""width="" height="">
</p>

#### Figure 1: Formula for calculating memory usage (%)

![](pictures/figure1.png)

# Data Preprocessing/Filtering

In order for the data to match our criteria for inputs and outputs, we performed a series of filters onto the raw input data. Because our goal was to predict future CPU utilization based on past resource utilization trends, we need to preprocess the data to represent it in this manner. We first assigned VMIDs to every virtual machines having the same no. of cores. Then, we created a new column Memory Usage % using the data from the column Memory usage [KB] .

To formulate the output for the dataset, we took the CPU usage % and performed classification using MLPclassifier.They were classified as Overutilization (90% and above) and Underutilization (20% and below). Then , we again performed classification in the same way but now ith the column Memory Usage %.

![](pictures/table3and4.png)

### Figure 2: CPU utilization (Over and Under) based on CPU usage[%]

<p align="left">
<img src="CPU_Utilization_based_on_CPU_usage.png?raw=true"
  alt=""width="200" height="250">
</p>

### Figure 3: CPU utilization (Over and Under) based on Memory usage[%]

<p align="left">
<img src="CPU_Utilization_based_on_Memory_usage.png?raw=true"
  alt=""width="200" height="250">
</p>

# Results

We have tested our two models on uniform classified data. For each type, we provide three different formats of input data: _cpu_, where each time step only contains CPU utilization; _cpu_mem_, where each time step has both CPU and memory utilization; and _all_, where each time step has all the data including CPU, memory, disk utilization and network traffic.

Table 5 below shows the train and testing accuracy for the neural network models. We can see that for the model, if we provide more information on each time step, the accuracy gets worse. This concludes that the extra information, i.e., memory, disk utilization and network traffic, act as noises for CPU utilization prediction. In addition, our modified AR-NN does not perform better than the default NN. This shows that the default NN is powerful enough for this kind of problems.

![](pictures/table5.png)

<p align="left">
<img src="Test_accuracy(CPU & Memory).png?raw=true"
  alt=""width="600" height="200">
</p>

**References**

- https://www.geeksforgeeks.org/libraries-in-python
- https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/learn/lecture/13756370
- https://www.sciencedirect.com/topics/earth-and-planetary-sciences/artificial-neural-network
- https://www.geeksforgeeks.org/artificial-neural-networks-and-its-applications
- https://stackoverflow.com/questions/45562078/ann-assignment-in-python3
