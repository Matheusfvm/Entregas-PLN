from sklearn.feature_extraction.text import CountVectorizer

from preprocess import preprocess_text
from predict_intent import predict_intent

vocabulary = {
  "qa_pairs": [
    {
      "pattern": "Como instalo o programa XYZ?",
      "response": "1. Acesse o site oficial do XYZ.\n2. Clique em “Download” ou “Baixar” na seção correspondente ao seu sistema operacional.\n3. Aguarde o término do download e execute o arquivo instalador.\n4. Siga as instruções do assistente de instalação (aceite o contrato de licença, escolha pasta de destino etc.).\n5. Ao final, clique em “Concluir” e abra o programa clicando no ícone criado na área de trabalho ou no menu de aplicativos."
    },
    {
      "pattern": "Quero instalar o XYZ. O que eu preciso fazer?",
      "response": "1. Verifique se seu sistema atende aos requisitos mínimos (memória, espaço em disco e versão do SO).\n2. Navegue até a página de downloads do XYZ dentro da aplicação web ou site oficial.\n3. Selecione a versão correta para Windows/Mac/Linux e baixe o instalador.\n4. Execute o instalador baixado e siga as telas de configuração padrão (próximo → aceitar termos → concluir).\n5. Após a instalação, reinicie a aplicação (se necessário) e verifique se o XYZ aparece no menu de “Meus Programas”."
    },
    {
      "pattern": "Preciso baixar e instalar o programa XYZ. Como faço passo a passo?",
      "response": "1. Entre na seção “Programas” ou “Downloads” da aplicação web.\n2. Localize o XYZ na lista de softwares disponíveis.\n3. Clique em “Baixar XYZ” e aguarde o download completar.\n4. Abra o instalador clicando duas vezes no arquivo baixado.\n5. Na tela de instalação, aceite os termos, escolha pasta de destino e finalize clicando em “Instalar”."
    },
    {
      "pattern": "Tem algum tutorial para instalar o software XYZ?",
      "response": "1. Sim. Dentro da aplicação web, acesse “Ajuda” → “Tutoriais”.\n2. Procure por “Instalação do XYZ” na lista de artigos.\n3. Siga as instruções detalhadas: baixar o instalador, executar como administrador e concluir configurações.\n4. Caso não encontre o tutorial, clique em “Enviar feedback” para solicitá-lo ou consulte a documentação PDF disponível em “Recursos”."
    },
    {
      "pattern": "Estou com erro ao instalar o programa XYZ. O que pode ser?",
      "response": "1. Verifique se o seu sistema operacional está atualizado (Windows Update, por exemplo).\n2. Confira se há espaço livre em disco suficiente (mínimo 500 MB).\n3. Veja se você tem permissões de administrador para executar o instalador.\n4. Caso o erro persista, acesse “Suporte” → “Erros de Instalação” na aplicação web e procure o código do erro ou envie um relatório para nossa equipe."
    },
    {
      "pattern": "Como acesso o painel de configurações?",
      "response": "1. Após fazer login, clique no ícone de engrenagem localizado no canto superior direito da tela.\n2. No menu suspenso, selecione “Configurações” para abrir o painel principal.\n3. Dentro do painel, você verá abas como “Conta”, “Preferências” e “Segurança”. Clique na aba desejada para ajustar opções específicas."
    },
    {
      "pattern": "Como faço para chegar na página de ajuda?",
      "response": "1. No canto inferior direito da aplicação, há um ícone “?” ou “Ajuda”.\n2. Clique nesse ícone para abrir o painel de ajuda rápida.\n3. Se precisar de mais detalhes, selecione “Central de Ajuda” para navegar pela documentação completa."
    },
    {
      "pattern": "Quero ver minhas notificações. Onde fica essa seção?",
      "response": "1. No topo da tela, à direita, há um sino (ícone de notificações).\n2. Clique no sino para abrir a lista das notificações mais recentes.\n3. Para ver todas, clique em “Ver todas as notificações” no rodapé desse menu suspenso."
    },
    {
      "pattern": "Como navego até o módulo de relatórios avançados?",
      "response": "1. Primeiro, clique em “Relatórios” na barra lateral esquerda.\n2. Em seguida, escolha a opção “Avançados” que aparece abaixo de “Relatórios Básicos”.\n3. Caso não veja “Avançados”, verifique se sua conta possui permissão para acessar esse módulo ou entre em “Configurações → Permissões” para habilitar."
    },
    {
      "pattern": "Como crio uma nova conta?",
      "response": "1. Acesse a página inicial da aplicação web e clique em “Cadastrar-se” ou “Criar Conta”.\n2. Preencha o formulário com nome, e-mail, senha e confirme a senha.\n3. Clique em “Registrar” e aguarde a mensagem de confirmação enviada por e-mail.\n4. Abra o e-mail enviado, clique no link de ativação e pronto: sua conta estará ativa."
    },
    {
      "pattern": "Como altero minha senha?",
      "response": "1. Faça login e clique no seu avatar ou nome de usuário no canto superior direito.\n2. Selecione “Configurações de Conta” → “Segurança” ou “Senha”.\n3. Informe sua senha atual, depois digite a nova senha duas vezes.\n4. Clique em “Salvar alterações”. Você verá uma mensagem confirmando a troca."
    },
    {
      "pattern": "Onde edito meus dados pessoais?",
      "response": "1. Após entrar na sua conta, vá em “Perfil” (ícone de usuário ou “Meu Perfil”).\n2. Clique em “Editar Perfil” ou “Atualizar Informações”.\n3. Altere campos como nome, telefone, data de nascimento e clique em “Salvar”.\n4. Caso queira mudar sua foto de perfil, clique em “Alterar Foto” e faça o upload de uma nova imagem."
    },
    {
      "pattern": "Como adiciono um endereço de e-mail secundário à minha conta?",
      "response": "1. No painel de “Configurações de Conta”, vá até “E-mails” ou “Contatos”.\n2. Clique em “Adicionar Novo E-mail” e digite o endereço secundário.\n3. Um e-mail de verificação será enviado para o novo endereço.\n4. Abra o e-mail, clique no link de confirmação e o e-mail adicional será ativado."
    },
    {
      "pattern": "Como excluo minha conta/perfil?",
      "response": "1. Entre em “Configurações de Conta” → “Segurança” (ou “Privacidade”).\n2. Role a tela até o final e clique em “Excluir Conta”.\n3. Confirme sua senha para autenticar a ação.\n4. Clique em “Confirmar exclusão”. Atenção: essa ação é irreversível."
    }
  ]
}


patterns = []
pair_responses = []
for pair in vocabulary['qa_pairs']:
    patterns.append(preprocess_text(pair['pattern']))
    pair_responses.append(pair['response'])

vectorizer = CountVectorizer()
pattern_vectors = vectorizer.fit_transform(patterns).toarray()

def chatbot(user_input, threshold):
     
     return predict_intent(
        user_input,
        threshold,
        vectorizer,
        pattern_vectors,
        pair_responses
    )

exemplo2 = "Onde encontro o relatório?"
print(f"Intenção prevista (Similaridade) para '{exemplo2}': {chatbot(exemplo2, 0.5)}")