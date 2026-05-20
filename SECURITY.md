# Política de Segurança

O Maracatu é um projeto de pesquisa em LLMs com pesos abertos. Levamos a sério tanto a segurança do código quanto questões de risco do modelo (privacidade, viés, geração de conteúdo perigoso). Obrigado por nos ajudar a manter o projeto seguro e responsável.

## Reportando uma vulnerabilidade ou problema de modelo

**Não abra issues públicas para vulnerabilidades.** Em vez disso:

- Use [GitHub Security Advisories](https://github.com/maracatu-labs/maracatu/security/advisories/new) (preferencial — privado por padrão), **ou**
- Envie e-mail para **contact@maracatu.org** com:
  - Descrição do problema
  - Passos para reproduzir
  - Impacto potencial
  - Sugestão de mitigação, se tiver

Vamos confirmar o recebimento em até 72 horas e trabalhar com você para entender e corrigir o problema. Após a correção ser publicada, podemos creditar você na nota de lançamento (se desejar).

## Escopo

### Código (segurança tradicional)

- Pipeline de treino (`src/maracatu/`)
- Scripts de preparação de corpus e deploy (`scripts/`)
- Configurações (`configs/`)
- Tokenizer (`tokenizer/`)

Exemplos: injeção de código via input não sanitizado, deserialização insegura de checkpoints, leakage de credenciais em logs.

### Modelo (segurança responsável)

- Geração de conteúdo claramente perigoso (instruções para violência, autoharm, etc.)
- Vazamento detectável de dados do corpus de treino (memorização)
- Viés sistemático grave (racismo, sexismo, conteúdo extremista) que escapa do baseline esperado de um modelo pré-treinado sem instruction tuning

**Importante:** Maracatu é um modelo base (sem RLHF / instruction tuning até o 800M). Comportamentos típicos de modelo base — repetição, geração inconsistente, ausência de filtros — não são vulnerabilidades. Documentamos esse escopo no [MODEL_CARD.md](MODEL_CARD.md).

### Fora de escopo

- Vulnerabilidades em dependências de terceiros (PyTorch, transformers, datasets) — reporte ao mantenedor original primeiro. Avise-nos se afetar o Maracatu diretamente.
- Saídas inesperadas que ficam dentro do comportamento esperado de um modelo base PT-BR.
- Ataques que dependem de acesso físico ou social engineering contra contribuidores específicos.

## Boas práticas para quem usa os pesos

- **Não use Maracatu em produção crítica sem fine-tuning e safety layer apropriados.** Os modelos publicados são pré-treinados (base models), sem alinhamento.
- Audite saídas antes de exibi-las a usuários finais.
- Considere quantização e inferência local para reduzir superfície de ataque (vazamento de inputs sensíveis para APIs externas).
- Para uso comercial, leia a [Apache 2.0](LICENSE) — você pode, mas é responsável pelo uso.

## Histórico de avisos

Quando publicarmos um aviso de segurança, ele aparecerá em [GitHub Security Advisories](https://github.com/maracatu-labs/maracatu/security/advisories).
