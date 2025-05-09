from screener_agent.models.response_schema import ScreeningResponseSchema
import json

if __name__ == '__main__':
    schema_dict = ScreeningResponseSchema.model_json_schema()
    json_output = json.dumps(schema_dict, indent=2)
    print(json_output)
