# ML-Schizophrenia
Informe Técnico: Metodología del Modelado Predictivo en Esquizofrenia (Criterios 2026)
La esquizofrenia es un trastorno mental grave y altamente heterogéneo cuya etiopatogenia se explica principalmente a través de la hipótesis del neurodesarrollo. Esta teoría postula que una susceptibilidad genómica subyacente interactúa de manera diacrónica y no lineal con insultos ambientales acumulativos durante ventanas críticas del desarrollo, alterando la maduración de los circuitos cerebrales prefrontales mucho antes de la aparición clínica de los síntomas primarios en la juventud. El presente informe documenta con rigor científico la metodología aplicada para traducir la literatura científica acumulada entre 2020 y 2026 en un framework computacional predictivo. Este flujo abarca desde el diseño lógico de un conjunto de datos clínicos sintéticos interactivos hasta la implementación de un pipeline de Machine Learning especializado en la clasificación e interpretabilidad de fenómenos biomédicos de baja prevalencia.
1. Enfoque Traslacional y Justificación de los Datos Sintéticos
En el ámbito de la psiquiatría computacional y la bioinformática, el acceso a registros médicos electrónicos o biobancos reales de pacientes con psicosis se encuentra fuertemente limitado por estrictas restricciones éticas, legales y de confidencialidad (como el Reglamento General de Protección de Datos). Asimismo, la esquizofrenia cuenta con una prevalencia global relativamente baja, estimada entre el 1% y el 3% en poblaciones de riesgo, lo que genera un sesgo extremo de desbalanceo de clases que dificulta el entrenamiento inicial de algoritmos predictivos sin un marco metodológico previo.
Para superar estas barreras y validar la viabilidad de un modelo clasificador no lineal, se optó por una estrategia de modelado traslacional: la creación de un dataset sintético basado en reglas clínicas estrictas extraídas de la literatura médica contemporánea. Este enfoque no pretende sustituir un ensayo clínico, sino actuar como un "sandbox" metodológico o prueba de concepto. Permite simular interacciones moleculares y socioambientales complejas, calibrar la respuesta del modelo ante el desequilibrio epidemiológico y garantizar la reproducibilidad matemática de las hipótesis etiológicas antes de comprometer recursos en biobancos masivos reales.
2. Arquitectura del Dataset y Especificación de Variables
Se diseñó una matriz de datos tabulares integrada por 5,000 observaciones (sujetos ficticios) y 12 dimensiones biológicas, ambientales y conductuales. Las variables independientes recopiladas abarcan desde el periodo prenatal hasta el debut psicótico en la juventud tardía, estructurándose de la siguiente manera:
Nombre de la Variable
Tipo de Dato
Rango / Codificación
Sustento Epistemológico e Interacción
 
Sexo_Varón
Binario
0: Mujer | 1: Hombre
Simula el ratio epidemiológico de tercera generación (3 hombres por cada 2 mujeres) y actúa como modulador del dimorfismo sexual placentario.
Edad_Paterna
Entero
20 a 55 años
Factor de riesgo epigenético germinal. Edades superiores a 45 y 50 años correlacionan exponencialmente con mutaciones de novo y alteraciones en la metilación del gen BDNF en el esperma.
Historial_Familiar
Binario
0: No | 1: Sí
Refleja la carga hereditaria clásica. Siguiendo datos poblacionales, se calibra para que aproximadamente el 94% de la muestra carezca de antecedentes directos, demostrando que la genética aislada es insuficiente.
PlacGRS
Continuo
0.0 a 1.0 (Normal)
Puntuación de Riesgo Poligénico Placentario. Resume los SNP expresados diferencialmente en el tejido placentario bajo estrés intrauterino agudo.
Complicaciones_Obstetricas
Binario
0: No | 1: Sí
Eventos perinatales severos (hipoxia, preeclampsia o bajo peso al nacer). Activa la vulnerabilidad destructiva latente del PlacGRS.
Deficiencia_Vitamina_D
Binario
0: No | 1: Sí
Modelado en función de la estacionalidad (nacimientos en invierno/primavera), afectando la inmunorregulación mucosa a través de la vía del receptor AhR.
Translocacion_LPS_Eje_Gut
Continuo
0.0 a 1.0
Nivel plasmático de lipopolisacáridos bacterianos derivado de la permeabilidad intestinal aberrante. Actúa como la primera señal de cebado (priming) inmune central.
Derrota_Social_HPA
Continuo
0.0 a 1.0
Estrés psicosocial crónico (migración, crianza urbana, aislamiento). Genera hipercortisolemia y desensibilización central de receptores de glucocorticoides.
Abuso_Infantil_Trauma
Binario
0: No | 1: Sí
Trauma temprano grave. Sensibiliza de forma permanente las vías dopaminérgicas mesolímbicas a través de la metilación epigenética del gen NR3C1.
Potencia_THC_Consumo
Ordinal
0: No | 1: Mod | 2: Alto
Uso de cannabis exógeno en la adolescencia. El nivel 2 representa cepas híbridas de alta potencia (15-25% de THC) que provocan el down-regulation del receptor CB1.
Edad_Inicio_Cannabis
Entero
14 a 20 años (0 si no)
Variable temporal crítica. Si el inicio diario ocurre antes de los 16 años, precipita la desinhibición de glutamato prefrontal e interrumpe la poda sináptica normal.
Diagnostico_Esquizofrenia
Binario
0: Control | 1: Caso
Variable Objetivo (Target). Determinada mediante la interceptación probabilística de las funciones de interacción molecular.

3. Marco Matemático de Interacción y Generación de Riesgo
Para alejarse de los modelos predictivos lineales aditivos tradicionales, que fracasan al capturar la complejidad biológica, la variable objetivo se calculó inyectando interacciones cruzadas no lineales basadas en las vías biológicas descubiertas en el periodo 2020-2026. La probabilidad individual de transición a psicosis ($P_{riesgo}$) se define mediante la función logística sigmoide:
P_{riesgo} = \frac{1}{1 + e^{-(\beta_0 + \text{Interacción}_{\text{Placenta}} + \text{Señal}_{\text{NLRP3}} + \text{Efecto}_{\text{EPA}} + \text{Efecto}_{\text{Cannabis}} + \text{Efecto}_{\text{Trauma}})}}
Las funciones internas de riesgo se parametrizaron bajo los siguientes criterios algorítmicos:
Sinergia Genético-Placentaria con Dimorfismo Sexual: Se implementó la regla $\text{Interacción}_{\text{Placenta}} = \text{PlacGRS} \times \text{Complicaciones}_{\text{Obstetricas}} \times \text{Sexo}_{\text{Varón}} \times 2.5$. Esto simula matemáticamente el hallazgo de que un alto riesgo poligénico placentario reduce el volumen cortical neonatal exclusivamente ante partos patológicos, afectando selectivamente a los fetos masculinos.
Modelo de Doble Señal Inmune (Inflamasa NLRP3): La permeabilidad de las barreras mucosa y hematoencefálica se simuló mediante la interacción $\text{Señal}_{\text{NLRP3}} = \text{Translocacion}_{\text{LPS}} \times \text{Derrota}_{\text{Social}} \times 3.0$. Esto recrea el mecanismo donde los lipopolisacáridos circulantes realizan el cebado microglial (Señal 1) y el cortisol derivado del estrés crónico detona la activación del inflamasa NLRP3 (Señal 2), provocando una fagocitosis sináptica destructiva.
Aceleración del Fenotipo Exógeno (THC): El consumo diario de cannabis de alta potencia aportó un peso crítico directo ($\beta = 5.0$). Para los registros donde el diagnóstico final se estableció en 1, se aplicó una penalización temporal sobre la variable secundaria Edad_Debut_FEP, reduciendo la edad media de aparición del primer brote psicótico entre 2 y 4 años si el inicio de consumo ocurrió antes de los 16 años, replicando la vulnerabilidad de la poda prefrontal adolescente.
4. Pipeline de Machine Learning y Clasificación No Lineal
Una vez estructurado el dataset en la memoria del entorno de ejecución, se desarrolló el pipeline computacional en Python empleando la librería Scikit-Learn y el algoritmo de gradiente impulsado XGBoost. El diseño del pipeline se rigió bajo los estándares de la ingeniería de características clínica:
División Estratificada del Espacio de Datos: Ante la baja prevalencia del target, una división aleatoria convencional podría alterar las proporciones de control/caso en los conjuntos de evaluación. Se aplicó una partición 80/20 forzando una estratificación estricta (stratify=y), garantizando que tanto el set de entrenamiento como el de prueba preserven la tasa exacta de desbalanceo epidemiológico.
Compensación Activa del Desbalanceo de Clases: Para evitar que el optimizador del algoritmo ignore la clase minoritaria (pacientes con esquizofrenia) y maximice sesgadamente la precisión global a costa de falsos negativos, se calculó dinámicamente el parámetro scale_pos_weight mediante el ratio de casos negativos sobre positivos ($\text{Casos}_{\text{Neg}} / \text{Casos}_{\text{Pos}}$). Este peso penaliza matemáticamente al algoritmo por cada error cometido al omitir un caso positivo.
Optimización del Optimizador Matemático: Se sustituyó la función de pérdida estándar por la optimización directa del Área bajo la Curva Precision-Recall (eval_metric='aucpr'). A diferencia de la curva ROC, la curva PR se enfoca exclusivamente en la precisión de la clase minoritaria, aislando el rendimiento real del modelo frente a padecimientos de baja prevalencia.
5. Protocolo de Validación y Explicabilidad (SHAP)
La validación del modelo omitió deliberadamente la métrica de 'Accuracy' (Exactitud), reconociéndola como un indicador falaz en conjuntos biomédicos desbalanceados. En su lugar, se desplegó un panel de evaluación multidimensional que incluye el reporte analítico de Precision, Recall y puntuación F1, respaldado por una matriz de confusión para auditar los errores de diagnóstico de tipo I y tipo II.
Para resolver el problema de "caja negra" intrínseco en modelos basados en árboles complejos como XGBoost, se integró un pipeline de Inteligencia Artificial Explicable (XAI) utilizando la teoría de juegos cooperativos a través de la librería SHAP (SHapley Additive exPlanations). Mediante el uso de un estimador TreeExplainer, se calcularon los valores SHAP para el conjunto de prueba, traduciendo las fronteras de decisión abstractas del algoritmo en impactos marginales específicos por variable. El despliegue final del SHAP Summary Plot ordena jerárquicamente las variables según su poder predictivo global y mapea cromáticamente (puntos rojos y azules) cómo los valores continuos elevados de translocación inmunológica intestinal, disfunción del eje HPA y exposición al THC actúan como vectores de fuerza que desplazan la predicción matemática hacia la confirmación clínica del brote psicótico.
