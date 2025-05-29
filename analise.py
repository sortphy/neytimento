import nltk
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.classify import NaiveBayesClassifier
from collections import Counter
import string

# Download dos recursos necessários do NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ClassificadorSentimentos:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.classificador = None
        
        # Conjunto de frases de treinamento (positivas e negativas)
        self.frases_treinamento = [
            # Frases positivas
            ("I love this movie, it's fantastic!", 'positivo'),
            ("This is an amazing product, highly recommend it!", 'positivo'),
            ("Great service and excellent quality!", 'positivo'),
            ("I'm so happy with this purchase!", 'positivo'),
            ("Wonderful experience, will definitely come back!", 'positivo'),
            ("Best restaurant in town, delicious food!", 'positivo'),
            ("This book is incredible, couldn't put it down!", 'positivo'),
            ("Outstanding performance, truly impressive!", 'positivo'),
            ("Love the design and functionality!", 'positivo'),
            ("Perfect solution for my needs!", 'positivo'),
            ("Excellent customer support, very helpful!", 'positivo'),
            ("This made my day, thank you!", 'positivo'),
            
            # Frases negativas
            ("I hate this product, it's terrible!", 'negativo'),
            ("Worst experience ever, completely disappointed!", 'negativo'),
            ("This is garbage, waste of money!", 'negativo'),
            ("Horrible service, very rude staff!", 'negativo'),
            ("I regret buying this, doesn't work at all!", 'negativo'),
            ("Awful quality, broke after one day!", 'negativo'),
            ("This movie is boring and pointless!", 'negativo'),
            ("Terrible food, couldn't even finish it!", 'negativo'),
            ("Disappointing results, not worth it!", 'negativo'),
            ("Bad design, very uncomfortable to use!", 'negativo'),
            ("Poor customer service, no help at all!", 'negativo'),
            ("This ruined my day, very frustrating!", 'negativo'),
        ]
    
    def preprocessar_texto(self, texto):
        """
        Preprocessa o texto removendo pontuação, convertendo para minúsculas,
        removendo stop words e aplicando stemming.
        """
        # Converter para minúsculas
        texto = texto.lower()
        
        # Remover pontuação
        texto = texto.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenizar
        tokens = word_tokenize(texto)
        
        # Remover stop words e aplicar stemming
        tokens_processados = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return tokens_processados
    
    def extrair_caracteristicas(self, tokens):
        """
        Extrai características do texto para o classificador.
        Retorna um dicionário com as características.
        """
        # Conta a frequência das palavras
        contador_palavras = Counter(tokens)
        
        # Cria características baseadas na presença de palavras
        caracteristicas = {}
        for palavra in contador_palavras:
            caracteristicas[f'contains({palavra})'] = True
            
        # Adiciona características sobre o comprimento
        caracteristicas['num_palavras'] = len(tokens)
        caracteristicas['texto_longo'] = len(tokens) > 10
        
        return caracteristicas
    
    def treinar_classificador(self):
        """
        Treina o classificador Naive Bayes com as frases de exemplo.
        """
        print("Iniciando treinamento do classificador...")
        
        # Preprocessar e extrair características de todas as frases
        dados_treinamento = []
        for frase, sentimento in self.frases_treinamento:
            tokens = self.preprocessar_texto(frase)
            caracteristicas = self.extrair_caracteristicas(tokens)
            dados_treinamento.append((caracteristicas, sentimento))
        
        # Embaralhar os dados
        random.shuffle(dados_treinamento)
        
        # Treinar o classificador
        self.classificador = NaiveBayesClassifier.train(dados_treinamento)
        
        print(f"Treinamento concluído com {len(dados_treinamento)} exemplos!")
        
        # Mostrar as características mais informativas
        print("\nCaracterísticas mais informativas:")
        self.classificador.show_most_informative_features(10)
    
    def classificar_sentimento(self, texto):
        """
        Classifica o sentimento de um texto como positivo ou negativo.
        """
        if self.classificador is None:
            raise ValueError("Classificador não foi treinado ainda!")
        
        # Preprocessar o texto
        tokens = self.preprocessar_texto(texto)
        caracteristicas = self.extrair_caracteristicas(tokens)
        
        # Classificar
        sentimento = self.classificador.classify(caracteristicas)
        
        # Obter probabilidades
        prob_dist = self.classificador.prob_classify(caracteristicas)
        confianca = prob_dist.prob(sentimento)
        
        return sentimento, confianca
    
    def avaliar_classificador(self):
        """
        Avalia o classificador usando validação cruzada simples.
        """
        if self.classificador is None:
            raise ValueError("Classificador não foi treinado ainda!")
        
        print("\n" + "="*50)
        print("AVALIAÇÃO DO CLASSIFICADOR")
        print("="*50)
        
        # Frases de teste (não usadas no treinamento)
        frases_teste = [
            ("This is absolutely wonderful!", 'positivo'),
            ("I'm really disappointed with this", 'negativo'),
            ("Great job, keep it up!", 'positivo'),
            ("This is the worst thing ever", 'negativo'),
            ("Pretty good, I like it", 'positivo'),
            ("Not impressed, could be better", 'negativo'),
        ]
        
        acertos = 0
        total = len(frases_teste)
        
        for frase, sentimento_real in frases_teste:
            sentimento_pred, confianca = self.classificar_sentimento(frase)
            acerto = sentimento_pred == sentimento_real
            acertos += acerto
            
            status = "✓" if acerto else "✗"
            print(f"{status} '{frase[:50]}...' -> {sentimento_pred} ({confianca:.2f}) [Real: {sentimento_real}]")
        
        precisao = acertos / total
        print(f"\nPrecisão: {acertos}/{total} = {precisao:.2%}")

def main():
    """
    Função principal para demonstrar o uso do classificador.
    """
    print("CLASSIFICADOR DE SENTIMENTOS COM NLTK")
    print("="*50)
    
    # Criar e treinar o classificador
    classificador = ClassificadorSentimentos()
    classificador.treinar_classificador()
    
    # Avaliar o classificador
    classificador.avaliar_classificador()
    
    # Demonstração interativa
    print("\n" + "="*50)
    print("DEMONSTRAÇÃO INTERATIVA")
    print("="*50)
    
    frases_exemplo = [
        "I absolutely love this new phone!",
        "This movie was terrible and boring",
        "The food at this restaurant is amazing",
        "I hate waiting in long lines",
        "What a beautiful day today!",
        "This software is full of bugs",
    ]
    
    print("Testando frases de exemplo:")
    for frase in frases_exemplo:
        sentimento, confianca = classificador.classificar_sentimento(frase)
        emoji = "😊" if sentimento == 'positivo' else "😞"
        print(f"{emoji} '{frase}' -> {sentimento.upper()} (confiança: {confianca:.2%})")
    
    print("\n" + "="*50)
    print("Agora você pode testar suas próprias frases!")
    print("Digite 'sair' para encerrar o programa.")
    print("="*50)
    
    while True:
        try:
            frase_usuario = input("\nDigite uma frase para classificar: ").strip()
            
            if frase_usuario.lower() in ['sair', 'exit', 'quit']:
                print("Encerrando o programa. Obrigado!")
                break
            
            if not frase_usuario:
                print("Por favor, digite uma frase válida.")
                continue
            
            sentimento, confianca = classificador.classificar_sentimento(frase_usuario)
            emoji = "😊" if sentimento == 'positivo' else "😞"
            
            print(f"\nResultado: {emoji} {sentimento.upper()}")
            print(f"Confiança: {confianca:.2%}")
            
            if confianca < 0.6:
                print("⚠️  Atenção: Baixa confiança na classificação!")
            
        except KeyboardInterrupt:
            print("\n\nPrograma interrompido pelo usuário.")
            break
        except Exception as e:
            print(f"Erro: {e}")

if __name__ == "__main__":
    main()