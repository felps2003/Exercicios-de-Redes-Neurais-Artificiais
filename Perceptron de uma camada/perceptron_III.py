import numpy as np

#Operador logico AND
#entradas = np.array([[0,0],[0,1],[1,0],[1,1]])
#saidas = np.array([0,0,0,1])
#Operador logico OR
entradas = np.array([[0,0],[0,1],[1,0],[1,1]])
saidas = np.array([0,1,1,1])

pesos = np.array([0.0, 0.0])
taxaDeAprendizagem = 0.1



def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0

def calculaSaida(registro):
    s = registro.dot(pesos)
    return stepFunction(s)


# Essa função atualiza os pesos realizando os calculos exigidos
def treino():
    erroTotal = 1
    while erroTotal != 0:
        erroTotal = 0
        for i in range(len(saidas)):
            saidaCalculada = calculaSaida(np.asarray(entradas[i]))
            erro = saidas[i] - saidaCalculada # Fazendo o calculo do erro, que pega a saida esperada menos o resultado da saida calculada
            erroTotal += erro
            for j in range(len(pesos)):
                pesos[j] = pesos[j] + (taxaDeAprendizagem * entradas[i][j] * erro) # aqui estamos atualizando os pesos fazendo o calculo exigido 
                #p(n+1) = p + (taxa de aprendizagem * entrada * erro calculado)
                print(f"Peso atualizado: {pesos[j]}")
            print(f"Total de erros: {erroTotal}")
            print("")

treino()
print("Rede Neural Treinada")
print(f"Suas saidas são {calculaSaida(entradas[0])}, {calculaSaida(entradas[1])}, {calculaSaida(entradas[2])}, {calculaSaida(entradas[3])}")