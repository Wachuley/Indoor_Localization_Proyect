This is a repository for my proyect on the localization of a device using data from different sensors, like accelerometers and gyroscopes for my for my automatic learning class

Project by Alexis Raúl Reyes Castillo

INSTRUCTIONS:
STEP 1: Install libraries
From inside the proyect folder, run:
python -m pip install pandas numpy scikit-learn matplotlib joblib odfpy openpyxl

STEP 2: Prepare the data
Execute preparar_datos.py indicating the files with the sensor data, intervals and map
python preparar_datos.py --sens datos/sensores.csv --ts datos/tiempos.csv --map datos/mapa.ods --out .
The result of this instruction is datos_entrenamiento.csv.

STEP 3: Train the models
This runs the models and generates the train data in the results folder. run:
python entrenar_modelos.py

STEP 4: Generate the graphs
This will generate all the graphs using the results from the previous step, to better visualize the output of the models. run:
python generar_evidencias.py --datos datos_entrenamiento.csv --predicciones resultados/predicciones_prueba.csv
