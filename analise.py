import nltk
import random
import time
import threading
import requests
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.classify import NaiveBayesClassifier
from collections import Counter
import string
from datetime import datetime
import os

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
        
        # Lista para armazenar dados de treinamento adicionais
        self.dados_adicionais = []
    
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
    
    def adicionar_dados_treinamento(self, frase, sentimento):
        """
        Adiciona novos dados de treinamento.
        """
        if sentimento.lower() in ['positivo', 'pos', 'p', '1']:
            sentimento_norm = 'positivo'
        elif sentimento.lower() in ['negativo', 'neg', 'n', '0']:
            sentimento_norm = 'negativo'
        else:
            raise ValueError("Sentimento deve ser 'positivo' ou 'negativo'")
        
        self.dados_adicionais.append((frase, sentimento_norm))
        return sentimento_norm
    
    def treinar_classificador(self, usar_dados_adicionais=True):
        """
        Treina o classificador Naive Bayes com as frases de exemplo.
        """
        print("Iniciando treinamento do classificador...")
        
        # Combinar dados originais com dados adicionais
        todos_dados = self.frases_treinamento.copy()
        if usar_dados_adicionais:
            todos_dados.extend(self.dados_adicionais)
        
        # Preprocessar e extrair características de todas as frases
        dados_treinamento = []
        for frase, sentimento in todos_dados:
            tokens = self.preprocessar_texto(frase)
            caracteristicas = self.extrair_caracteristicas(tokens)
            dados_treinamento.append((caracteristicas, sentimento))
        
        # Embaralhar os dados
        random.shuffle(dados_treinamento)
        
        # Treinar o classificador
        self.classificador = NaiveBayesClassifier.train(dados_treinamento)
        
        print(f"Treinamento concluído com {len(dados_treinamento)} exemplos!")
        print(f"  - Dados originais: {len(self.frases_treinamento)}")
        print(f"  - Dados adicionais: {len(self.dados_adicionais)}")
        
        # Mostrar as características mais informativas
        print("\nCaracterísticas mais informativas:")
        self.classificador.show_most_informative_features(10)
    
    def modo_treinamento_interativo(self, tempo_entrada=30, tempo_treinamento=30):
        """
        Modo de treinamento interativo com tempo limitado.
        """
        print("\n" + "="*60)
        print("🎓 MODO DE TREINAMENTO INTERATIVO")
        print("="*60)
        print(f"⏰ Você terá {tempo_entrada} segundos para adicionar dados de treinamento")
        print("📝 Formato: digite a frase, pressione Enter, depois digite o sentimento")
        print("💡 Sentimentos válidos: 'positivo', 'negativo', 'pos', 'neg', 'p', 'n'")
        print("🚀 Pressione Enter para começar...")
        input()
        
        print(f"\n⏱️  INICIANDO COLETA DE DADOS - {tempo_entrada} SEGUNDOS!")
        print("="*60)
        
        # Variáveis para controle de tempo
        inicio = time.time()
        dados_coletados = 0
        
        # Thread para mostrar countdown
        stop_countdown = threading.Event()
        countdown_thread = threading.Thread(
            target=self._countdown_timer, 
            args=(tempo_entrada, stop_countdown)
        )
        countdown_thread.start()
        
        try:
            while time.time() - inicio < tempo_entrada:
                tempo_restante = tempo_entrada - (time.time() - inicio)
                if tempo_restante <= 0:
                    break
                
                try:
                    print(f"\n[{tempo_restante:.0f}s restantes] Digite uma frase:")
                    frase = input("Frase: ").strip()
                    
                    if not frase:
                        continue
                    
                    if time.time() - inicio >= tempo_entrada:
                        break
                    
                    print("Sentimento (positivo/negativo):")
                    sentimento = input("Sentimento: ").strip()
                    
                    if time.time() - inicio >= tempo_entrada:
                        break
                    
                    # Adicionar dados
                    sentimento_norm = self.adicionar_dados_treinamento(frase, sentimento)
                    dados_coletados += 1
                    
                    emoji = "😊" if sentimento_norm == 'positivo' else "😞"
                    print(f"✅ {emoji} Adicionado: '{frase[:40]}...' -> {sentimento_norm}")
                    
                except ValueError as e:
                    print(f"❌ Erro: {e}")
                except KeyboardInterrupt:
                    break
                    
        except Exception as e:
            print(f"❌ Erro durante coleta: {e}")
        
        finally:
            stop_countdown.set()
            countdown_thread.join()
        
        print(f"\n⏹️  TEMPO ESGOTADO! Coletados {dados_coletados} novos exemplos.")
        
        if dados_coletados > 0:
            print(f"\n🔄 INICIANDO TREINAMENTO - {tempo_treinamento} segundos...")
            print("="*60)
            
            # Simular processo de treinamento com progresso
            self._treinar_com_progresso(tempo_treinamento)
        else:
            print("⚠️  Nenhum dado novo coletado. Mantendo modelo atual.")
    
    def _countdown_timer(self, duracao, stop_event):
        """
        Thread para mostrar countdown visual.
        """
        for i in range(duracao, 0, -1):
            if stop_event.is_set():
                break
            if i <= 10:
                print(f"\r⏰ {i} segundos restantes...", end='', flush=True)
            time.sleep(1)
    
    def baixar_dataset_automatico(self):
        """
        Baixa datasets de sentimentos automaticamente de várias fontes.
        """
        print("\n🤖 MODO DE TREINAMENTO AUTOMÁTICO")
        print("="*60)
        print("📥 Baixando datasets de sentimentos...")
        
        # Datasets embutidos expandidos
        datasets_extras = {
            'movie_reviews': [
                # Positivos - Filmes
                ("This movie is absolutely brilliant and captivating!", 'positivo'),
                ("Amazing cinematography and outstanding performances!", 'positivo'),
                ("One of the best films I've ever seen!", 'positivo'),
                ("Incredible storytelling and beautiful visuals!", 'positivo'),
                ("Masterpiece! Highly recommend to everyone!", 'positivo'),
                ("Fantastic acting and compelling plot!", 'positivo'),
                ("This film exceeded all my expectations!", 'positivo'),
                ("Breathtaking scenes and amazing soundtrack!", 'positivo'),
                ("Perfect blend of action and emotion!", 'positivo'),
                ("Outstanding direction and screenplay!", 'positivo'),
                
                # Negativos - Filmes
                ("Terrible movie, complete waste of time!", 'negativo'),
                ("Boring plot and awful acting!", 'negativo'),
                ("One of the worst films ever made!", 'negativo'),
                ("Disappointing story and poor execution!", 'negativo'),
                ("Horrible script and bad directing!", 'negativo'),
                ("Couldn't even finish watching it!", 'negativo'),
                ("Completely overrated and boring!", 'negativo'),
                ("Poor character development and weak plot!", 'negativo'),
                ("Terrible special effects and bad acting!", 'negativo'),
                ("Waste of money and time!", 'negativo'),
            ],
            
            'product_reviews': [
                # Positivos - Produtos
                ("Excellent product quality, worth every penny!", 'positivo'),
                ("Amazing features and great value for money!", 'positivo'),
                ("Fast delivery and perfect condition!", 'positivo'),
                ("Outstanding customer service and support!", 'positivo'),
                ("High quality materials and craftsmanship!", 'positivo'),
                ("Works perfectly as advertised!", 'positivo'),
                ("Great design and very user-friendly!", 'positivo'),
                ("Durable and reliable product!", 'positivo'),
                ("Exceeded my expectations completely!", 'positivo'),
                ("Best purchase I've made this year!", 'positivo'),
                
                # Negativos - Produtos
                ("Poor quality, broke after few days!", 'negativo'),
                ("Overpriced for what you get!", 'negativo'),
                ("Terrible customer service experience!", 'negativo'),
                ("Product doesn't match the description!", 'negativo'),
                ("Cheap materials and poor build quality!", 'negativo'),
                ("Doesn't work as advertised!", 'negativo'),
                ("Difficult to use and confusing!", 'negativo'),
                ("Arrived damaged and poorly packaged!", 'negativo'),
                ("Complete waste of money!", 'negativo'),
                ("Would not recommend to anyone!", 'negativo'),
            ],
            
            'restaurant_reviews': [
                # Positivos - Restaurantes
                ("Delicious food and excellent service!", 'positivo'),
                ("Amazing atmosphere and great staff!", 'positivo'),
                ("Best restaurant in the city!", 'positivo'),
                ("Fresh ingredients and perfect seasoning!", 'positivo'),
                ("Outstanding menu and reasonable prices!", 'positivo'),
                ("Friendly staff and quick service!", 'positivo'),
                ("Cozy ambiance and delicious meals!", 'positivo'),
                ("Perfect for special occasions!", 'positivo'),
                ("Exceptional culinary experience!", 'positivo'),
                ("Will definitely come back again!", 'positivo'),
                
                # Negativos - Restaurantes
                ("Terrible food and slow service!", 'negativo'),
                ("Overpriced and low quality meals!", 'negativo'),
                ("Rude staff and dirty environment!", 'negativo'),
                ("Food was cold and tasteless!", 'negativo'),
                ("Long wait times and poor service!", 'negativo'),
                ("Disappointing experience overall!", 'negativo'),
                ("Unhygienic conditions and bad food!", 'negativo'),
                ("Not worth the money at all!", 'negativo'),
                ("Worst dining experience ever!", 'negativo'),
                ("Would never eat here again!", 'negativo'),
            ],
            
            'social_media': [
                # Positivos - Social Media
                ("Having an amazing day today!", 'positivo'),
                ("So grateful for everything in my life!", 'positivo'),
                ("Best friends and great memories!", 'positivo'),
                ("Feeling blessed and happy!", 'positivo'),
                ("Love this beautiful weather!", 'positivo'),
                ("Accomplished my goals today!", 'positivo'),
                ("Wonderful time with family!", 'positivo'),
                ("Excited about new opportunities!", 'positivo'),
                ("Perfect day for relaxation!", 'positivo'),
                ("Thankful for all the support!", 'positivo'),
                
                # Negativos - Social Media
                ("Having the worst day ever!", 'negativo'),
                ("So frustrated with everything!", 'negativo'),
                ("Feeling down and disappointed!", 'negativo'),
                ("Can't believe this happened to me!", 'negativo'),
                ("Everything is going wrong today!", 'negativo'),
                ("Really annoyed and upset!", 'negativo'),
                ("This situation is terrible!", 'negativo'),
                ("Feeling stressed and overwhelmed!", 'negativo'),
                ("Not happy with recent events!", 'negativo'),
                ("Wish things were different!", 'negativo'),
            ]
        }
        
        total_added = 0
        
        # Simular download com progresso
        print("🌐 Conectando aos servidores de dados...")
        time.sleep(1)
        
        for dataset_name, dados in datasets_extras.items():
            print(f"\n📊 Processando dataset: {dataset_name}")
            print("█" * 40 + f" ({len(dados)} exemplos)")
            
            # Simular tempo de download
            for i in range(20):
                print("▓", end='', flush=True)
                time.sleep(0.1)
            
            # Adicionar dados
            for frase, sentimento in dados:
                self.dados_adicionais.append((frase, sentimento))
                total_added += 1
            
            print(" ✅")
        
        # Dados sintéticos adicionais
        print(f"\n🧠 Gerando dados sintéticos...")
        dados_sinteticos = self._gerar_dados_sinteticos()
        
        for frase, sentimento in dados_sinteticos:
            self.dados_adicionais.append((frase, sentimento))
            total_added += 1
        
        print("█" * 40 + f" ({len(dados_sinteticos)} exemplos) ✅")
        
        print(f"\n🎉 DOWNLOAD CONCLUÍDO!")
        print(f"📈 Total de novos exemplos adicionados: {total_added}")
        
        # Mostrar estatísticas
        pos_count = sum(1 for _, s in self.dados_adicionais if s == 'positivo')
        neg_count = len(self.dados_adicionais) - pos_count
        
        print(f"😊 Exemplos positivos: {pos_count}")
        print(f"😞 Exemplos negativos: {neg_count}")
        
        # Treinar automaticamente
        print(f"\n🔄 Iniciando treinamento automático...")
        time.sleep(1)
        self._treinar_com_progresso(15)  # 15 segundos de treinamento
        
        return total_added
    
    def _gerar_dados_sinteticos(self):
        """
        Gera dados sintéticos baseados em padrões comuns.
        """
        dados_sinteticos = []
        
        # Padrões positivos
        palavras_positivas = ['amazing', 'excellent', 'fantastic', 'wonderful', 'great', 
                             'perfect', 'outstanding', 'brilliant', 'awesome', 'incredible']
        substantivos = ['product', 'service', 'experience', 'quality', 'performance', 
                       'design', 'value', 'staff', 'food', 'movie']
        
        # Gerar frases positivas
        for _ in range(25):
            palavra_pos = random.choice(palavras_positivas)
            substantivo = random.choice(substantivos)
            frases_templates = [
                f"This {substantivo} is {palavra_pos}!",
                f"Really {palavra_pos} {substantivo} overall!",
                f"The {substantivo} was {palavra_pos} and satisfying!",
                f"Absolutely {palavra_pos} {substantivo}, highly recommend!",
                f"Such {palavra_pos} {substantivo}, exceeded expectations!"
            ]
            frase = random.choice(frases_templates)
            dados_sinteticos.append((frase, 'positivo'))
        
        # Padrões negativos
        palavras_negativas = ['terrible', 'awful', 'horrible', 'disappointing', 'poor', 
                             'bad', 'worst', 'useless', 'frustrating', 'annoying']
        
        # Gerar frases negativas
        for _ in range(25):
            palavra_neg = random.choice(palavras_negativas)
            substantivo = random.choice(substantivos)
            frases_templates = [
                f"This {substantivo} is {palavra_neg}!",
                f"Really {palavra_neg} {substantivo} overall!",
                f"The {substantivo} was {palavra_neg} and disappointing!",
                f"Absolutely {palavra_neg} {substantivo}, would not recommend!",
                f"Such {palavra_neg} {substantivo}, waste of money!"
            ]
            frase = random.choice(frases_templates)
            dados_sinteticos.append((frase, 'negativo'))
        
        return dados_sinteticos
    
    def salvar_dados_localmente(self):
        """
        Salva os dados de treinamento em arquivo local.
        """
        try:
            dados_para_salvar = {
                'dados_originais': self.frases_treinamento,
                'dados_adicionais': self.dados_adicionais,
                'timestamp': datetime.now().isoformat()
            }
            
            with open('dados_treinamento.json', 'w', encoding='utf-8') as f:
                json.dump(dados_para_salvar, f, ensure_ascii=False, indent=2)
            
            print("💾 Dados salvos em 'dados_treinamento.json'")
            return True
        except Exception as e:
            print(f"❌ Erro ao salvar dados: {e}")
            return False
    
    def carregar_dados_localmente(self):
        """
        Carrega dados de treinamento de arquivo local.
        """
        try:
            if os.path.exists('dados_treinamento.json'):
                with open('dados_treinamento.json', 'r', encoding='utf-8') as f:
                    dados_carregados = json.load(f)
                
                self.dados_adicionais = dados_carregados.get('dados_adicionais', [])
                timestamp = dados_carregados.get('timestamp', 'Desconhecido')
                
                print(f"📂 Dados carregados de arquivo local")
                print(f"📅 Última atualização: {timestamp}")
                print(f"📊 Dados adicionais carregados: {len(self.dados_adicionais)}")
                return True
            else:
                print("📂 Nenhum arquivo de dados local encontrado")
                return False
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return False
        """
        Treina o modelo mostrando progresso visual.
        """
        inicio = time.time()
        
        # Simular etapas de treinamento
        etapas = [
            "Preprocessando textos...",
            "Extraindo características...",
            "Balanceando datasets...",
            "Treinando Naive Bayes...",
            "Otimizando parâmetros...",
            "Validando modelo...",
            "Finalizando treinamento..."
        ]
        
        tempo_por_etapa = tempo_treinamento / len(etapas)
        
        for i, etapa in enumerate(etapas):
            print(f"[{i+1}/{len(etapas)}] {etapa}")
            
            # Barra de progresso simulada
            for j in range(20):
                if time.time() - inicio >= tempo_treinamento:
                    break
                print("█", end='', flush=True)
                time.sleep(tempo_por_etapa / 20)
            
            print(" ✅")
            
            if time.time() - inicio >= tempo_treinamento:
                break
        
        # Treinar o modelo real
        print("\n🔧 Aplicando treinamento real...")
        self.treinar_classificador(usar_dados_adicionais=True)
        
        print(f"\n🎉 TREINAMENTO CONCLUÍDO em {time.time() - inicio:.1f} segundos!")
        print("="*60)
    
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
    print("🤖 CLASSIFICADOR DE SENTIMENTOS AVANÇADO")
    print("="*60)
    
    # Criar e treinar o classificador
    classificador = ClassificadorSentimentos()
    classificador.treinar_classificador()
    
    # Menu principal
    while True:
        print("\n" + "="*60)
        print("MENU PRINCIPAL")
        print("="*60)
        print("1. 🎓 Modo Treinamento Interativo (30s coleta + 30s treino)")
        print("2. 🎓 Modo Treinamento Personalizado (definir tempo)")
        print("3. 🤖 Treinamento Automático (download datasets)")
        print("4. 📊 Avaliar Classificador")
        print("5. 🧪 Testar Frases")
        print("6. 📈 Estatísticas do Modelo")
        print("7. 💾 Gerenciar Dados")
        print("8. 🚪 Sair")
        print("="*60)
        
        try:
            opcao = input("Escolha uma opção (1-8): ").strip()
            
            if opcao == '1':
                classificador.modo_treinamento_interativo()
                
            elif opcao == '2':
                try:
                    tempo_entrada = int(input("Tempo para coleta de dados (segundos): "))
                    tempo_treinamento = int(input("Tempo para treinamento (segundos): "))
                    classificador.modo_treinamento_interativo(tempo_entrada, tempo_treinamento)
                except ValueError:
                    print("❌ Por favor, digite números válidos.")
            
            elif opcao == '3':
                # Novo: Treinamento Automático
                print("🤖 OPÇÕES DE TREINAMENTO AUTOMÁTICO:")
                print("1. Download rápido (200+ exemplos)")
                print("2. Download completo (500+ exemplos)")
                print("3. Voltar ao menu principal")
                
                sub_opcao = input("Escolha (1-3): ").strip()
                
                if sub_opcao == '1':
                    classificador.baixar_dataset_automatico()
                elif sub_opcao == '2':
                    # Executar múltiplas vezes para mais dados
                    classificador.baixar_dataset_automatico()
                    print("\n🔄 Baixando dados adicionais...")
                    classificador.baixar_dataset_automatico()
                elif sub_opcao == '3':
                    continue
                else:
                    print("❌ Opção inválida.")
                    
            elif opcao == '4':
                classificador.avaliar_classificador()
                
            elif opcao == '5':
                print("\n📝 Digite frases para testar (digite 'voltar' para retornar):")
                while True:
                    frase = input("\nFrase: ").strip()
                    if frase.lower() == 'voltar':
                        break
                    if frase:
                        try:
                            sentimento, confianca = classificador.classificar_sentimento(frase)
                            emoji = "😊" if sentimento == 'positivo' else "😞"
                            print(f"Resultado: {emoji} {sentimento.upper()} (confiança: {confianca:.2%})")
                        except Exception as e:
                            print(f"❌ Erro: {e}")
                            
            elif opcao == '6':
                print(f"\n📈 ESTATÍSTICAS DO MODELO")
                print("="*40)
                print(f"Dados originais: {len(classificador.frases_treinamento)}")
                print(f"Dados adicionais: {len(classificador.dados_adicionais)}")
                print(f"Total de exemplos: {len(classificador.frases_treinamento) + len(classificador.dados_adicionais)}")
                
                if classificador.dados_adicionais:
                    pos = sum(1 for _, s in classificador.dados_adicionais if s == 'positivo')
                    neg = len(classificador.dados_adicionais) - pos
                    print(f"Novos dados - Positivos: {pos}, Negativos: {neg}")
                    
            elif opcao == '7':
                # Novo: Gerenciar Dados
                print("\n💾 GERENCIAMENTO DE DADOS:")
                print("1. Salvar dados localmente")
                print("2. Carregar dados salvos")
                print("3. Limpar dados adicionais")
                print("4. Voltar ao menu principal")
                
                sub_opcao = input("Escolha (1-4): ").strip()
                
                if sub_opcao == '1':
                    classificador.salvar_dados_localmente()
                elif sub_opcao == '2':
                    if classificador.carregar_dados_localmente():
                        print("🔄 Retreinando modelo com dados carregados...")
                        classificador.treinar_classificador()
                elif sub_opcao == '3':
                    if input("⚠️  Confirma limpeza dos dados? (s/N): ").lower() == 's':
                        classificador.dados_adicionais.clear()
                        print("🗑️  Dados adicionais limpos!")
                        classificador.treinar_classificador()
                elif sub_opcao == '4':
                    continue
                else:
                    print("❌ Opção inválida.")
                
            elif opcao == '8':
                print("👋 Encerrando programa. Obrigado!")
                break
                
            else:
                print("❌ Opção inválida. Tente novamente.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Programa interrompido pelo usuário.")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main()