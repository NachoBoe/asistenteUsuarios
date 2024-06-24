
from langchain.prompts import PromptTemplate




## ELEGIR UNA SECCION A PARTIR DE RESUMENES
prompt_indice_resumido = PromptTemplate(template=""" 
Dado la pregunta delimitada por <>, y una lista de secciones con un resumen de cada una delimitado por []. Identificar que seccion del indice de contenidos es la más relevante para responder la pregunta.
Obligatoriammente seguir estos pasos para llegar a la respuesta:

1. Comprende profundamente la pregunta: Analiza la pregunta para captar su esencia y el tipo de información que busca.

2. Examina las secciones: Revisa las descripciones de las secciones para encontrar cual mejor se alineen con la pregunta.

3. Reflexiona sobre la conexión entre la pregunta y el contenido: Decidir cual seccion es la más fuertemente relacionada con la pregunta, y cual de sus subsecciones y subsecciones hijas de estas son relevantes para responder la pregunta.

4. Respuesta Final: Una vez decidido que seccion es la más relevante a la pregunta, responder con un JSON BLOB, de la forma {{"n_seccion": Aca va el numero de seccion, "seccion":Aca va el nombre de la seccion}}

Pregunta: <{pregunta}>
Secciones: 
[
{contenido}
]
                                        
1. Comprende profundamente la pregunta: 

""",input_variables=["pregunta", "contenido"])



