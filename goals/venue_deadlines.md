# Deadlines de venue — consulta na fonte oficial

Tarefa 0.1 do `goals/roadmap_icml_ijcnn.md`. **Só entram aqui datas lidas no site
oficial da conferência.** Agregadores (mldeadlines, OpenCurious, ResearchTheta)
não são fonte; quando forem a única coisa disponível, o campo fica `PENDENTE`.

Re-executar semanalmente enquanto houver `PENDENTE`.

---

## IJCNN 2027 — CONFIRMADO

**Consulta:** 2026-09-29
**Fonte:** `https://www.ijcnn.org/2027/authors/call-for-papers` e as páginas de
evento abaixo (dados estruturados `schema.org/Event` embutidos na página).

| item | data | fonte |
|---|---|---|
| Special Sessions / Workshop proposal | 2026-11-15 | `/2027/events/special-sessions-and-workshop-proposal-deadline` |
| **Regular papers deadline** | **2027-01-31** | `/2027/events/regular-papers-deadline` (`startDate: 2027-01-31`) |
| Notificação | 2027-03-15 | `/2027/events/regular-papers-acceptance-notification` |
| Camera-ready (todos os tracks) | 2027-04-12 | `/2027/events/camera-ready-submissions-all-tracks` |
| Conferência | 2027-06-14 a 2027-06-18, Cape Town | `schema.org/Event` do site |

**Formato, citação literal da CFP:**

> "Prospective authors are invited to submit complete papers of no more than six
> (6) pages in the IEEE two-column conference proceedings format."

**Sistema de submissão:** Microsoft CMT (declarado na própria CFP).

**Ressalva:** a CFP não declara fuso horário do deadline nesta página. O
roadmap assume 23:59 UTC-12 com base em agregador — **isso continua não
verificado**. Tratar 2027-01-31 00:00 UTC-3 como data-limite de trabalho até
que a CFP publique o fuso.

---

## ICML 2027 — PENDENTE

**Consulta:** 2026-09-29

| URL | HTTP | resultado |
|---|---|---|
| `https://icml.cc/Conferences/2027/CallForPapers` | 404 | página não existe |
| `https://icml.cc/Conferences/2027` | 404 | página não existe |
| `https://icml.cc/` | 200 | só menciona ICML 2026 (19 ocorrências); zero menções a 2027 |

**Status:** o CFP do ICML 2027 **não foi publicado**. Nenhuma data oficial
existe neste momento.

**O que NÃO é fato:** as datas de ~16/01 (abstract) e ~22/01 ou ~28/01 (paper)
que circulam em agregadores. Os dois agregadores consultados em 29/09 discordam
entre si em 6 dias. Não usar em planejamento como se fossem confirmadas.

**Regra de planejamento enquanto pendente** (risco R1 do roadmap): planejar para
a data mais cedo citada, **22/01/2027**, e tratá-la como estimativa conservadora
e não como deadline. Re-checar semanalmente.

### Comando de re-checagem

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
for u in "https://icml.cc/Conferences/2027/CallForPapers" "https://icml.cc/Conferences/2027"; do
  echo "=== $u"
  curl -sL --max-time 25 -A "Mozilla/5.0" "$u" -o /tmp/icml.html -w "HTTP=%{http_code}\n"
  sed -e 's/<[^>]*>/\n/g' /tmp/icml.html | sed 's/^[ \t]*//' | grep -vE '^$' \
    | grep -iE "deadline|abstract|notification|january|february" | head -20
done
```

Saída atual: `HTTP=404` nos dois. Quando subir, gravar as datas aqui com a data
da consulta e substituir o status `PENDENTE`.

---

## Log de consultas

| data | ICML 2027 | IJCNN 2027 |
|---|---|---|
| 2026-09-29 | 404, CFP não publicado | confirmado, 31/01/2027, 6 pgs IEEE, CMT |
