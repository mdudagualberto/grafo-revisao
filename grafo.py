from typing import Literal, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

# Os eixos da revisão: o nome curto vira a categoria, a descrição vai para o prompt.
EIXOS = {
    "eixo 1": "AI4SE com SDD para orquestração de agentes, incluindo orquestração bidirecional",
    "eixo 2": "AI4SE para refatoração de sistemas legados com grafos de conhecimento",
    "eixo 3": "Métricas de produtividade de agentes AI4SE (incluindo LLMOps) e benchmarks de agentes",
}


# O "schema" da classificação: o eixo só pode ser um dos três.
class Classificacao(BaseModel):
    eixo: Literal["eixo 1", "eixo 2", "eixo 3"]
    justificativa: str = Field(description="Uma ou duas frases explicando a escolha.")


# Modelo principal e reserva: se o principal falhar (ex.: 503), usa o reserva.
llm_principal = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0, timeout=60, max_retries=3)
llm_reserva = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", timeout=60, max_retries=3)

llm = llm_principal.with_fallbacks([llm_reserva])
classificador = llm_principal.with_structured_output(Classificacao).with_fallbacks(
    [llm_reserva.with_structured_output(Classificacao)]
)


class Estado(TypedDict, total=False):
    abstract: str
    resumo: str
    eixo: str
    justificativa: str
    decisao_humana: str
    eixo_final: str


def resumir(estado: Estado) -> dict:
    prompt = (
        "Resuma o abstract abaixo em português, em no máximo três frases. "
        "Mantenha o problema, a abordagem e o principal resultado.\n\n"
        f"Abstract:\n{estado['abstract']}"
    )
    resposta = llm.invoke(prompt)
    conteudo = resposta.content
    # Modelos Gemini mais novos podem devolver uma lista de blocos: junta só o texto.
    if isinstance(conteudo, list):
        conteudo = "".join(bloco.get("text", "") for bloco in conteudo if isinstance(bloco, dict))
    return {"resumo": conteudo}


def classificar(estado: Estado) -> dict:
    lista_eixos = "\n".join(f"- {nome}: {descricao}" for nome, descricao in EIXOS.items())
    prompt = (
        "Você está fazendo a triagem de artigos de uma revisão sistemática. "
        "Classifique o artigo em exatamente um destes eixos:\n\n"
        f"{lista_eixos}\n\n"
        f"Resumo do artigo:\n{estado['resumo']}"
    )
    resultado = classificador.invoke(prompt)
    return {"eixo": resultado.eixo, "justificativa": resultado.justificativa}


def confirmar(estado: Estado) -> dict:
    pergunta = "Confirma o eixo? Digite 'ok' ou o eixo correto."
    # Pergunta de novo até receber 'ok' ou um eixo da lista.
    while True:
        resposta = interrupt(
            {
                "pergunta": pergunta,
                "resumo": estado["resumo"],
                "eixo_sugerido": estado["eixo"],
                "justificativa": estado["justificativa"],
                "opcoes": list(EIXOS),
            }
        )
        resposta = str(resposta).strip().lower()
        if resposta == "ok":
            return {"decisao_humana": "aprovado", "eixo_final": estado["eixo"]}
        if resposta in EIXOS:
            return {"decisao_humana": "corrigido", "eixo_final": resposta}
        pergunta = f"'{resposta}' não é um eixo válido. Digite 'ok' ou um destes: {', '.join(EIXOS)}."


builder = StateGraph(Estado)
builder.add_node("resumir", resumir)
builder.add_node("classificar", classificar)
builder.add_node("confirmar", confirmar)

builder.add_edge(START, "resumir")
builder.add_edge("resumir", "classificar")
builder.add_edge("classificar", "confirmar")
builder.add_edge("confirmar", END)

grafo = builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    abstract = (
        "Legacy systems accumulate technical debt that makes refactoring costly and risky. "
        "We propose an approach in which LLM-based agents build a knowledge graph of the "
        "codebase, linking modules, dependencies and business rules, and use it to plan "
        "incremental refactorings. In an industrial case study with a 400k-line COBOL system, "
        "the approach reduced the time to identify refactoring candidates by 60%."
    )
    config = {"configurable": {"thread_id": "artigo-1"}}

    # Primeira chamada: roda até o interrupt e para.
    grafo.invoke({"abstract": abstract}, config)

    estado = grafo.get_state(config)
    pergunta = estado.tasks[0].interrupts[0].value
    print("Parado antes de:", estado.next)
    print("\nResumo:", pergunta["resumo"])
    print("Eixo sugerido:", pergunta["eixo_sugerido"])
    print("Justificativa:", pergunta["justificativa"])

    # Retoma com o mesmo thread_id até o grafo terminar: se a resposta for inválida,
    # o confirmar pausa de novo com uma nova pergunta.
    while estado.next:
        pergunta = estado.tasks[0].interrupts[0].value
        resposta = input(f"\n{pergunta['pergunta']} ")
        final = grafo.invoke(Command(resume=resposta), config)
        estado = grafo.get_state(config)

    print("\nDecisão:", final["decisao_humana"], "| eixo final:", final["eixo_final"])