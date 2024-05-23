import PySimpleGUI as sg
import json

def get_schema():
    # Define the layout of the main window
    layout = [
        [sg.Text("Entity Name"), sg.InputText(key="entity_name")],
        [sg.Button("Add Property"), sg.Button("Save Entity"), sg.Button("Save Schema"), sg.Button("Exit")],
        [sg.Text("Properties:")],
        [sg.Listbox(values=[], size=(60, 10), key="properties")],
        [sg.Text("Primary Key"), sg.InputText(key="primary_key")],
        [sg.Text("Foreign Keys (col:ref_table.col)"), sg.InputText(key="foreign_keys")],
    ]

    # Create the main window
    window = sg.Window("Schema Input", layout)

    # Initialize schema dictionary and other variables
    schema = {"entities": {}}
    current_properties = []
    current_entity = ""

    while True:
        # Read events and values from the window
        event, values = window.read()
        
        # Exit the loop if the window is closed or Exit button is clicked
        if event == sg.WINDOW_CLOSED or event == "Exit":
            break

        # Handle the Add Property button click
        if event == "Add Property":
            # Define the layout of the property input window
            prop_layout = [
                [sg.Text("Property Name"), sg.InputText(key="property_name")],
                [sg.Text("Type"), sg.Combo(["int", "float", "string", "enum", "date"], key="property_type")],
                [sg.Text("Min (for int/float/date)"), sg.InputText(key="min_value")],
                [sg.Text("Max (for int/float/date)"), sg.InputText(key="max_value")],
                [sg.Text("Values (for enum, comma separated)"), sg.InputText(key="enum_values")],
                [sg.Text("Starting Characters (for string, comma separated)"), sg.InputText(key="starting_chars")],
                [sg.Button("Save Property"), sg.Button("Cancel")],
            ]

            # Create the property input window
            prop_window = sg.Window("Add Property", prop_layout)

            while True:
                # Read events and values from the property input window
                prop_event, prop_values = prop_window.read()
                
                # Close the property input window if the window is closed or Cancel button is clicked
                if prop_event == sg.WINDOW_CLOSED or prop_event == "Cancel":
                    break
                
                # Handle the Save Property button click
                if prop_event == "Save Property":
                    # Get the property details from the input fields
                    prop_name = prop_values["property_name"]
                    prop_type = prop_values["property_type"]
                    property = {"type": prop_type}

                    # Handle int and float property types
                    if prop_type in ["int", "float"]:
                        try:
                            min_val = int(prop_values["min_value"]) if prop_type == "int" else float(prop_values["min_value"])
                            max_val = int(prop_values["max_value"]) if prop_type == "int" else float(prop_values["max_value"])
                            property["min"] = min_val
                            property["max"] = max_val
                        except ValueError:
                            sg.popup("Please enter valid numerical values for min and max.")
                            continue
                    # Handle date property type
                    elif prop_type == "date":
                        property["min"] = prop_values["min_value"]
                        property["max"] = prop_values["max_value"]
                    # Handle enum property type
                    if prop_type == "enum":
                        property["values"] = prop_values["enum_values"].split(",")
                    # Handle string property type
                    if prop_type == "string":
                        property["starting character"] = prop_values["starting_chars"].split(",")

                    # Add the property to the current properties list
                    current_properties.append((prop_name, property))
                    # Update the properties listbox in the main window
                    window["properties"].update([f"{prop_name}: {property}" for prop_name, property in current_properties])
                    break

            # Close the property input window
            prop_window.close()

        # Handle the Save Entity button click
        if event == "Save Entity":
            # Get the entity details from the input fields
            current_entity = values["entity_name"]
            entity_schema = {
                "properties": {name: prop for name, prop in current_properties},
                "primary_key": values["primary_key"],
                "foreign_keys": {
                    fk.split(":")[0]: fk.split(":")[1] for fk in values["foreign_keys"].split(",") if fk
                }
            }
            # Add the entity to the schema
            schema["entities"][current_entity] = entity_schema
            sg.popup(f"Entity {current_entity} saved.")
            # Reset the input fields and current properties list
            current_properties = []
            window["entity_name"].update("")
            window["primary_key"].update("")
            window["foreign_keys"].update("")
            window["properties"].update([])

        # Handle the Save Schema button click
        if event == "Save Schema":
            # Save the schema to a JSON file
            with open("schema.json", "w") as f:
                json.dump(schema, f, indent=4)
            sg.popup("Schema saved to schema.json")
            break

    # Close the main window
    window.close()
    return schema

if __name__ == "__main__":
    schema = get_schema()
    print(json.dumps(schema, indent=4))
