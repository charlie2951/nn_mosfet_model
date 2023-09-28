# Modeling of Inversion layer capacitance using ANN-based Regression Technique
![image](https://github.com/charlie2951/nn_mosfet_model/assets/90516512/b6a44032-0513-4ed7-ac61-74b40c561a97)

**ABSTRACT:** </br>
This work presents a data-driven regression model of inversion layer capacitance of double gate III-V channel MOSFETs implemented using an artificial neural network. The training dataset is generated using a Schrodinger-Poisson solver for different channel thicknesses, carrier effective masses, oxide thickness, barrier height, and a wide range of gate bias voltages. The neural network predicted capacitance value is compared with Schrodinger-Poisson solver data and a physics-based analytical model result. The model effectively captures the variation in channel thickness, barrier height, carrier effective mass, and oxide thickness. Furthermore, extensive error analysis has been performed to demonstrate the correctness and degree of accuracy of the predicted result.
</br></br>
**Training dataset:** *training_data_eigen.csv*</br>
**Test dataset:** *test01.csv to test10.csv*</br>
**Source code:** *Gate_capacitance_v02.py file*</br>
Pre-trained models are inside *trained_model* directory. The file name reflects the number of neurons used in each layer for example "16_16_8_3 " represents 4 layer network 
with 16 neurons in 1st and 2nd layer, 8 neurons in 3rd layer and 3 in the final output layer.</br>

If you are using this repository then ***Cite this work as***</br>
*Maity, S.K., Pandit, S. Modeling of inversion layer capacitance of III-V double gate MOSFETs using a neural network-based regression technique. J Comput Electron (2023). https://doi.org/10.1007/s10825-023-02089-7*
</br>
For more details refer to the above article.
