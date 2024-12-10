# Inteligência Artificial - IFPE

**Classificação do distúrbio do sono utilizando Aprendizado de Máquina**
Docente: Luciano de Souza Cabral
Discentes: Heitor Fidelis Beda Caete, Jose Joaquim de Santana Neto, Renan Freitas dos Anjos

### Resumo :dart:
Transtornos do sono, como insônia e apneia do sono, são os
principais fatores ligados a baixa qualidade do sono na população mundial. A
compreensão esses distúrbios, com ferramentas de machine learning, podem
ajudar a compreender o desvio padrão desses distúrbios e contribuem nas
análises detalhadas de variáveis individuais, que permitem explorar relações
causais entre essas características e distúrbios relacionados ao sono. O
presente estudo possui como foco central abordar o uso de machine learning,
utilizando método classificação como Gradient Boosting, Decision Tree e Cat
Boost, para analisar e classificar dados cruciais de saúde.

### Diretórios :file_folder:
- **modeling**: Contém os códigos de treinamento e teste dos modelos de machine learning, realizando uma comparação entre os modelos Decision Tree, Gradient Boosting e Cat Boost.
  > Como executar:
    ```bash
    $ cd modeling

    $ python3 -m venv venv

    $ chmod +x venv/bin/activate

    $ ./venv/bin/activate

    $ pip install -r requirements.txt

    $ python3 modeling.py
    ```

- **app**: Contém o código da aplicação web utilizando a lib streamlit, que permite a interação com o modelo utilizando Gradient Boosting.
    > Como executar:
    ```bash
    $ cd app

    $ python3 -m venv venv

    $ chmod +x venv/bin/activate

    $ ./venv/bin/activate

    $ pip install -r requirements.txt

    $ streamlit run app.py
    ```