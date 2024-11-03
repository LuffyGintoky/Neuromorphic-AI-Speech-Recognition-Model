import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import mu_0
import matplotlib.pyplot as plt
from scipy.constants import e
import math
from sklearn.metrics import accuracy_score
import csv
import sys
from tqdm import tqdm

# Cargar el subconjunto de datos de entrenamiento
pickle_file_path_train = r'/data/X_train.pickle' 
with open(pickle_file_path_train, 'rb') as file:
    X_train = pickle.load(file)

print("Datos de entrenamiento cargados correctamente")

# Cargar el subconjunto de datos de prueba
pickle_file_path_test = r'/data/X_test.pickle'
with open(pickle_file_path_test, 'rb') as file:
    X_test = pickle.load(file)

print("Datos de prueba cargados correctamente")

# Ruta al archivo pickle
file_path = r'/data/y_train.pickle'

# Cargar el archivo pickle
with open(file_path, 'rb') as file:
    Y_train = pickle.load(file)


# Ruta al archivo pickle
file_path = r'/data/y_test.pickle'

# Cargar el archivo pickle
with open(file_path, 'rb') as file:
    Y_test = pickle.load(file)

def process_AI(L,R,I_DC,rango_I_in,X_train,X_test,Y_train,Y_test,Neurons):

    c = 0.02
    gyro_ratio = -1.76e11 # Gyromagnetic ratio (1/Ts)
    gyro_ratio_hz = -2.8e10 # Gyroratio (Hz/T)
    alpha_g = 0.01 # Gilbert Damping Constant
    aplied_field = 0.400 # Applied Field (Tesla)
    saturation_magnetization = 0.73 # Saturation Magnetization (T)
    saturation_magnetization_A_M = saturation_magnetization / mu_0
    epsilon = 0.5 # Spin Polarization Efficiency
    lande_factor = -2.002 # Electron Lande Factor 
    bhor_magneton = 9.27e-24 # Bohr magneton (J/T)
    e = -1.602e-19 # Electron Charge (C)


    def masking(num_canales, num_neuronas, array_n0_ntau,random_matrix, duracion_ns=100, Delta_t=5e-9, plot=False):

        #array aleatorio binario
        array_aleatorio_binario = random_matrix

        # Realiza el producto
        procesed_input = array_n0_ntau @ array_aleatorio_binario

        # Aplanar el array resultante
        flattened_data = procesed_input.reshape(-1)

        # Calcular la cantidad de pasos a repetir cada valor para mantenerlo durante el periodo especificado
        pasos_por_valor = int((duracion_ns * 1e-9) / Delta_t)

        # Extender cada valor en el array durante el periodo especificado
        valores_extendidos = np.repeat(flattened_data, pasos_por_valor)

        # Crear un array de tiempo basado en la escala de tiempo y la duración
        tiempo = np.arange(len(valores_extendidos)) * Delta_t * 1e9  # Convertir a nanosegundos

        # Graficar
        if plot:
            plt.figure(figsize=(15, 5))
            plt.plot(tiempo, valores_extendidos, linestyle='-', color='purple')
            plt.title('Masking Preprocesed Input')
            plt.xlabel('Tiempo (ns)')
            plt.ylabel('Valor')
            
            plt.grid(True)
            plt.show()

        return valores_extendidos


    def scale_to_range(data, global_min , global_max , new_min=-2, new_max=2):
        old_min = global_min
        old_max = global_max
        old_range = old_max - old_min
        normalized_data = (data - old_min) / old_range
        scaled_data = normalized_data * (new_max - new_min) + new_min

        return scaled_data


    # Función para calcular I_th y T_relax

    def Compute_I_th_and_Relax_time(L, R, I_DC):
        omega_0 = gyro_ratio * (aplied_field -  saturation_magnetization)
        gama_g = alpha_g * omega_0
        sigma = (epsilon * lande_factor * bhor_magneton) / (2 * e * saturation_magnetization_A_M * L * (np.pi * (R ** 2)))
        I_th = gama_g / sigma
        T_relax = 1 / (2 * gama_g * ((I_DC / I_th) - 1))
        omega_M = -gyro_ratio*saturation_magnetization
        Q = (2*omega_M)/omega_0-1
        return T_relax, I_th , Q 


    def rango(longitud_total):
        # Dividir la longitud total entre 2 para encontrar los límites
        limite_inferior = -longitud_total // 2
        limite_superior = longitud_total // 2
        
        # Crear la tupla con los límites
        return (limite_inferior, limite_superior)



    num_canales = 13
    num_neuronas = Neurons
    random_matrix = np.random.choice([-1, 1], size=(num_canales, num_neuronas))
    N_steps = 5
    


    T_relax, I_th, Q  = Compute_I_th_and_Relax_time(L=L, R=R, I_DC=I_DC)

    thetha_ns = (T_relax/4) * 1e9

    delta_t = (thetha_ns * 1e-9)/N_steps

    masked_input_train = []

    total_processing_times = 0  # Para calcular el tiempo promedio

    for index, data in enumerate(X_train):
        start_time = time.time()  # Tiempo de inicio
        
        array_n0_ntau = data.T
        masked_values = masking(num_canales, num_neuronas, array_n0_ntau, random_matrix=random_matrix,duracion_ns=thetha_ns,Delta_t=delta_t, plot=False)
        masked_input_train.append(masked_values)
        
        end_time = time.time()  # Tiempo de finalización
        processing_time = end_time - start_time  # Tiempo total de procesamiento
        total_processing_times += processing_time

        # Calcular el tiempo promedio de procesamiento hasta ahora
        average_processing_time = total_processing_times / (index + 1)

        # Estimar el tiempo restante
        remaining_items = len(X_train) - (index + 1)
        estimated_time_remaining = average_processing_time * remaining_items        

    total_processing_times = 0  # Para calcular el tiempo promedio


    masked_input_test = []

    for index, data in enumerate(X_test):
        start_time = time.time()  # Tiempo de inicio
        
        array_n0_ntau = data.T
        masked_values = masking(num_canales, num_neuronas, array_n0_ntau,random_matrix=random_matrix, duracion_ns=thetha_ns,Delta_t=delta_t, plot=False)
        masked_input_test.append(masked_values)
        
        end_time = time.time()  # Tiempo de finalización
        processing_time = end_time - start_time  # Tiempo total de procesamiento
        total_processing_times += processing_time

        # Calcular el tiempo promedio de procesamiento hasta ahora
        average_processing_time = total_processing_times / (index + 1)

        # Estimar el tiempo restante
        remaining_items = len(X_test) - (index + 1)
        estimated_time_remaining = average_processing_time * remaining_items
        
    print(f"Masking Completed")

    masked_input_train_32 = []

    for input_data in masked_input_train:
        input_data = input_data.astype(np.float32)
        masked_input_train_32.append(input_data)


    masked_input_test_32 = []

    for input_data in masked_input_test:
        input_data = input_data.astype(np.float32)
        masked_input_test_32.append(input_data)


    train_array = np.concatenate(masked_input_train_32)

    # Calcula el mínimo y el máximo globales
    global_min = np.min(train_array)
    global_max = np.max(train_array)

    rango_ = rango(rango_I_in)


    Input_data_train = []

    total_processing_times = 0  # Para calcular el tiempo promedio
    for index , input in enumerate(masked_input_train_32):

        start_time = time.time()  # Tiempo de inicio

        scaled_time_series = scale_to_range(input,global_min=global_min,global_max=global_max,new_min=rango_[0],new_max=rango_[1])
        Input_data_train.append(scaled_time_series)

        end_time = time.time()  # Tiempo de finalización
        processing_time = end_time - start_time  # Tiempo total de procesamiento
        total_processing_times += processing_time

        # Calcular el tiempo promedio de procesamiento hasta ahora
        average_processing_time = total_processing_times / (index + 1)

        # Estimar el tiempo restante
        remaining_items = len(masked_input_train_32) - (index + 1)
        estimated_time_remaining = average_processing_time * remaining_items


        
    Input_data_test = []

    total_processing_times = 0  # Para calcular el tiempo promedio
    for index , input in enumerate(masked_input_test_32):

        start_time = time.time()  # Tiempo de inicio

        scaled_time_series = scale_to_range(input,global_min=global_min,global_max=global_max)
        Input_data_test.append(scaled_time_series)

        end_time = time.time()  # Tiempo de finalización
        processing_time = end_time - start_time  # Tiempo total de procesamiento
        total_processing_times += processing_time

        # Calcular el tiempo promedio de procesamiento hasta ahora
        average_processing_time = total_processing_times / (index + 1)

        # Estimar el tiempo restante
        remaining_items = len(masked_input_test_32) - (index + 1)
        estimated_time_remaining = average_processing_time * remaining_items


         
    print(f"Scaling Completed")   

    def round_up(numero, decimales=0):
        factor = 10 ** decimales
        return math.ceil(numero * factor) / factor

    I_th_corr = round_up(I_th, 4)

    I_corr = I_th_corr + rango_[1]*1e-3 - I_DC

    print(f"Parametros de la simulacion: Theta (Thetha): {thetha_ns:.4f} ns - Delta t: {delta_t:.4e} s - I_th: {I_th:.4f} A - I_DC: {I_DC:.4f} A - T_relax: {T_relax:.4e} s - I_corr: {I_corr:.4f} A - Rango: ({rango_[0]}, {rango_[1]}) mA")

    # Definición de la dinamica del oscilador
    def uniform_oscillator(Delta_t,I_in, v_osc_previous,T_relax,Q,I_c,I_corr,I_DC):
        
        if ((I_DC + I_corr + I_in)/(I_c) - 1 )/((I_DC + I_corr + I_in)/(I_c) + Q ) < 0 :
            argument = 0

        else: 
            argument = ((I_DC + I_corr + I_in)/(I_c) - 1 )/((I_DC + I_corr + I_in)/(I_c) + Q )
        
        v_infinity = c * np.sqrt(argument)
        v_osc =  v_infinity * (1 - np.exp(-Delta_t/T_relax)) + v_osc_previous * np.exp(-Delta_t/T_relax)
        return v_osc

    def uniform_oscilator_processing(time_series_I_in):
            
            # Inicializar la respuesta del oscilador
            osc_response = []

            # Valor inicial del oscilador 
            v_osc_previous = 0

            # Aplicar la función del oscilador a cada punto de la serie de tiempo
            for I_in in time_series_I_in:
                v_osc = uniform_oscillator(I_in=I_in,I_corr=I_corr,v_osc_previous=v_osc_previous,T_relax=T_relax,Q=Q,I_c=I_th,Delta_t=delta_t,I_DC=I_DC)
                osc_response.append(v_osc)
                v_osc_previous = v_osc  # Actualizar el estado anterior para la próxima iteración

            return osc_response


    reservoir_outputs_train = []
    total_processing_times = 0  # Para calcular el tiempo promedio



    for index, input_data in enumerate(tqdm(Input_data_train, desc="Training Progress")):


        # Condición para omitir el procesamiento y agregar solo ceros
        if I_DC < I_th:
            # Añadir una lista de ceros del mismo tamaño que la salida esperada
            output = [0] * len(input_data)
            reservoir_outputs_train.append(output)  # Asegúrate de que el tamaño de ceros sea correcto
        else:
            # Procesar la entrada actual solo si I_DC >= I_C
            output = uniform_oscilator_processing(input_data * 1e-3)
            reservoir_outputs_train.append(output)





    reservoir_outputs_train_sampled = [output[::5] for output in reservoir_outputs_train]
   # print("Forma de un elemento muestreado:", len(reservoir_outputs_train_sampled[3]))
    reservoir_outputs_train_transformed = [np.array(output).reshape(-1, Neurons).T for output in reservoir_outputs_train_sampled]
   # print("Forma de un elemento transformado en reservoir_outputs_train:", reservoir_outputs_train_transformed[3].shape)

    reservoir_outputs_test = []
    total_processing_times = 0  # Para calcular el tiempo promedio

    for index, input_data in enumerate(tqdm(Input_data_test, desc="Testing Progress")):

        # Condición para omitir el procesamiento y agregar solo ceros
        if I_DC < I_th:
            # Añadir una lista de ceros del mismo tamaño que la salida esperada
            output = [0] * len(input_data)
            reservoir_outputs_test.append(output)  # Asegúrate de que el tamaño de ceros sea correcto
            #print(output)
        else:
            # Procesar la entrada actual solo si I_DC >= I_C
            output = uniform_oscilator_processing(input_data * 1e-3)
            reservoir_outputs_test.append(output)






    reservoir_outputs_test_sampled = [output[::5] for output in reservoir_outputs_test]
    #print("Forma de un elemento muestreado:", len(reservoir_outputs_test_sampled[4]))
    reservoir_outputs_test_transformed = [np.array(output).reshape(-1, Neurons).T for output in reservoir_outputs_test_sampled]
    #print("Forma de un elemento transformado en reservoir_outputs_test:", reservoir_outputs_test_transformed[4].shape)


    def linear_classifier_readout_variable_length(X_train, y_train, num_features,X_test, y_test):
        
        def create_S_matrix(mfcc_features, num_features=13):

            # Flatten each MFCC feature array and concatenate
            transposed_features = [sequence.T for sequence in mfcc_features]
            S = np.vstack(transposed_features)


            return S

        S = create_S_matrix(X_train)

        def create_Y_hat_repeated(y_train, X_train, num_features):

            # Contar los intervalos en cada secuencia de X_train
            intervals_per_sample = [sequence.shape[1] for sequence in X_train]

            Y_hat_repeated = np.repeat(y_train, intervals_per_sample, axis=0)

            return Y_hat_repeated

        Y_hat_repeated = create_Y_hat_repeated(y_train, X_train, num_features)

        # Calculamos la pseudoinversa 

        S_moore = np.linalg.pinv(S)

        Weights = Y_hat_repeated.T @ S_moore.T

        # Imprimir los resultados
        #print('Shapes:')
        #print("Forma de la matriz S:", S.shape)
        #print("Y_hat_repeated shape:", Y_hat_repeated.shape)
        #print("Pseudo inverse shape :",S_moore.shape)
        #print('Weights matrix shape', Weights.shape)
        #print('Weigths:',Weights)
    
    # Aplicamos los pesos al conjunto de prueba
        
        def apply_weights_and_predict(data_set, Weights, num_features):


            # Initialize an empty list to store averaged results for each test sample
            averaged_results = []

            # Apply weights to each test sequence and compute the average result
            for sequence in data_set:

                # Apply weights to the sequence
                interval_results = sequence.T @ Weights.T
        
                # Average the results over all intervals for the test sample
                averaged_results.append(np.mean(interval_results, axis=0))

            # Convert the list of averages to a NumPy array
            averaged_results = np.array(averaged_results)

            # Predict the class for each test sample
            y_test_pred = np.argmax(averaged_results, axis=1)

            return y_test_pred , averaged_results
        
        
        y_test_pred , y_test_promedio = apply_weights_and_predict(X_test, Weights, num_features)
        y_train_pred , y_train_promedio = apply_weights_and_predict(X_train,Weights,num_features)

        # Convertimos las etiquetas verdaderas one-hot en índices de clase
        y_test_true_indices = np.argmax(y_test, axis=1)
        y_train_true_indices = np.argmax(y_train,axis=1)

        
        # Calculamos la precisión
        accuracy_test = accuracy_score(y_test_true_indices, y_test_pred)
        accuracy_train = accuracy_score(y_train_true_indices,y_train_pred)

        #print('Shapes for Test Set :')
        #print("Mean Results shape:", y_test_promedio.shape)
        #print('Predicted Test Labels:')
        #print(y_test_pred)
        #print('true Test Labels:')
        #print( y_test_true_indices)
        print(f'Accuracy en el conjunto de prueba: {accuracy_test*100:.2f} %')

        #print('Shapes for train Set :')
        #print("Mean Results shape:", y_train_promedio.shape)
        #print('Predicted train Labels:')
        #print(y_train_pred)
        print(f'Accuracy en el conjunto de entrenamiento: {accuracy_train*100:.2f} %')

        return accuracy_test , accuracy_train

    num_samples_test = 200
    num_samples_train = 1800
    num_labels = 10
    num_features = Neurons

    accuracy_test , accuracy_train = linear_classifier_readout_variable_length(
    reservoir_outputs_train_transformed, Y_train, num_features=num_features , X_test=
    reservoir_outputs_test_transformed, y_test=Y_test)

    return accuracy_test , accuracy_train , I_th , T_relax 

# Define your lists of parameters
L_list = np.linspace(2e-9,20e-9,19)  # L values in meters
R_list = np.linspace(50e-9,100e-9,21)  # R values in meters
IDC_list = [0.00275]  # I_DC_Vortex values in Amperes
rango_I_in_list = [4]  # rango_I_in values in mA
Neurons = 1000
# Nombre del archivo CSV
csv_filename = 'sim_2_75_results.csv'

if __name__ == "__main__":
    # Abrimos el archivo una vez para crear los headers
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['Model Number', 'L', 'R', 'I_DC_Vortex', 'rango_I_in', 'I_th', 'T_relax', 'Accuracy_Train', 'Accuracy_Test']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
    # Total de modelos a procesar
    total_models = len(L_list) * len(R_list) * len(IDC_list) * len(rango_I_in_list)
    model_number = 0
    total_start_time = time.time()

    # Loop through the parameter combinations
    for L in L_list:
        for R in R_list:
            for I_DC_Vortex in IDC_list:
                for rango_I_in in rango_I_in_list:
                    model_number += 1
                    start_time = time.time()
                    print(f"Training model {model_number}/{total_models} with L={L}, R={R}, I_DC_Vortex={I_DC_Vortex}, rango_I_in={rango_I_in}")

                    row = {
                        'Model Number': model_number,
                        'L': L,
                        'R': R,
                        'I_DC_Vortex': I_DC_Vortex,  
                        'rango_I_in': rango_I_in,
                        'I_th': None,
                        'T_relax': None,
                        'Accuracy_Train': None,
                        'Accuracy_Test': None
                    }

                    try:
                        # Call your process_AI function
                        accuracy_test, accuracy_train, I_th, T_relax = process_AI(
                        L, R, I_DC_Vortex, rango_I_in, X_train, X_test, Y_train, Y_test ,Neurons=Neurons
                        )

                        # Update row with results
                        row.update({
                            'I_th': I_th,
                            'T_relax': T_relax,
                            'Accuracy_Train': accuracy_train,
                            'Accuracy_Test': accuracy_test
                        })

                    except Exception as e:
                        print(f"An error occurred while processing model {model_number}: {e}")

                    # Open the CSV file in append mode for each model
                    with open(csv_filename, mode='a', newline='') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        writer.writerow(row)

                    # Calculate time taken and estimate remaining time
                    end_time = time.time()
                    model_time = end_time - start_time
                    elapsed_time = end_time - total_start_time
                    remaining_models = total_models - model_number
                    estimated_total_time = (elapsed_time / model_number) * total_models
                    estimated_remaining_time = estimated_total_time - elapsed_time

                    print(f"Model {model_number} completed in {model_time:.2f} seconds.")
                    print(f"Estimated remaining time: {estimated_remaining_time / 60:.2f} minutes.\n")

    print("All models have been processed.")

