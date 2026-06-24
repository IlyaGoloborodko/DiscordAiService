"""
Один входящий запрос из внешнего сервиса.

Поля:

id
external_request_id — id из другого сервиса
user_id
session_id / conversation_id
prompt_text
user_metadata JSONB — язык, страна, подписка, устройство, возрастной сегмент, и так далее
context_metadata JSONB — источник запроса, сценарий, UI-экран, A/B-variant
status — received / processing / done / failed
created_at, updated_at

Это ваш главный “хвост”, к которому потом всё привязывается.


Один запрос может породить несколько вызовов LLM, например retry, different model, tool calling.

Поля:

id
request_id
model_name
temperature
system_prompt_version
input_tokens
output_tokens
latency_ms
raw_response JSONB
final_text
finish_reason
created_at

Если у вас будет несколько этапов генерации, это очень удобно для отладки.

Инструменты / действия — tool_calls

Если LLM вызывает что-то вроде:

TTS
YouTube search
ранжирование
retrieval
перевод

Поля:

id
request_id
llm_run_id
tool_name
tool_version
input_payload JSONB
output_payload JSONB
status
error_code
latency_ms
created_at
"""