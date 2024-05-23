import PySimpleGUI as sg
import json

def get_schema():
    layout = [
        [sg.Text("Entity Name"), sg.InputText(key="entity_name")],
        [sg.Button("Add Property"), sg.Button("Save Entity"), sg.Button("Save Schema"), sg.Button("Exit")],
        [sg.Text("Properties:")],
        [sg.Listbox(values=[], size=(60, 10), key="properties")],
        [sg.Text("Primary Key"), sg.InputText(key="primary_key")],
        [sg.Text("Foreign Keys (col:ref_table.col)"), sg.InputText(key="foreign_keys")],
    ]

    window = sg.Window("Schema Input", layout)

    schema = {"entities": {}}
    current_properties = []
    current_entity = ""

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == "Exit":
            break

        if event == "Add Property":
            prop_layout = [
                [sg.Text("Property Name"), sg.InputText(key="property_name")],
                [sg.Text("Type"), sg.Combo(["int", "float", "string", "enum", "date"], key="property_type")],
                [sg.Text("Min (for int/float/date)"), sg.InputText(key="min_value")],
                [sg.Text("Max (for int/float/date)"), sg.InputText(key="max_value")],
                [sg.Text("Values (for enum, comma separated)"), sg.InputText(key="enum_values")],
                [sg.Text("Starting Characters (for string, comma separated)"), sg.InputText(key="starting_chars")],
                [sg.Button("Save Property"), sg.Button("Cancel")],
            ]

            prop_window = sg.Window("Add Property", prop_layout)

            while True:
                prop_event, prop_values = prop_window.read()
                if prop_event == sg.WINDOW_CLOSED or prop_event == "Cancel":
                    break
                if prop_event == "Save Property":
                    prop_name = prop_values["property_name"]
                    prop_type = prop_values["property_type"]
                    property = {"type": prop_type}

                    if prop_type in ["int", "float"]:
                        try:
                            min_val = int(prop_values["min_value"]) if prop_type == "int" else float(prop_values["min_value"])
                            max_val = int(prop_values["max_value"]) if prop_type == "int" else float(prop_values["max_value"])
                            property["min"] = min_val
                            property["max"] = max_val
                        except ValueError:
                            sg.popup("Please enter valid numerical values for min and max.")
                            continue
                    elif prop_type == "date":
                        property["min"] = prop_values["min_value"]
                        property["max"] = prop_values["max_value"]
                    if prop_type == "enum":
                        property["values"] = prop_values["enum_values"].split(",")
                    if prop_type == "string":
                        property["starting character"] = prop_values["starting_chars"].split(",")

                    current_properties.append((prop_name, property))
                    window["properties"].update([f"{prop_name}: {property}" for prop_name, property in current_properties])
                    break

            prop_window.close()

        if event == "Save Entity":
            current_entity = values["entity_name"]
            entity_schema = {
                "properties": {name: prop for name, prop in current_properties},
                "primary_key": values["primary_key"],
                "foreign_keys": {
                    fk.split(":")[0]: fk.split(":")[1] for fk in values["foreign_keys"].split(",") if fk
                }
            }
            schema["entities"][current_entity] = entity_schema
            sg.popup(f"Entity {current_entity} saved.")
            current_properties = []
            window["entity_name"].update("")
            window["primary_key"].update("")
            window["foreign_keys"].update("")
            window["properties"].update([])

        if event == "Save Schema":
            with open("schema.json", "w") as f:
                json.dump(schema, f, indent=4)
            sg.popup("Schema saved to schema.json")
            break

    window.close()
    return schema

if __name__ == "__main__":
    schema = get_schema()
    print(json.dumps(schema, indent=4))
