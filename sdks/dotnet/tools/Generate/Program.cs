// Writes src/GetStream.VisionAgents/Generated/Models.cs from acceleration/api/openapi.yaml.
//
//   dotnet run --project tools/Generate            regenerate
//   dotnet run --project tools/Generate -- --check fail if the committed file is stale
//
// The library rather than the nswag CLI, because two things the CLI cannot be told are what
// make the output usable: PascalCase property names, and enums read as strings.
using NJsonSchema;
using NJsonSchema.CodeGeneration;
using NJsonSchema.CodeGeneration.CSharp;
using NJsonSchema.Visitors;
using NSwag;
using NSwag.CodeGeneration.CSharp;

// Run from sdks/dotnet, which is where both paths are relative to.
var here = Directory.GetCurrentDirectory();
var spec = Path.GetFullPath(Path.Combine(here, "../../acceleration/api/openapi.yaml"));
var output = Path.GetFullPath(Path.Combine(here, "src/GetStream.VisionAgents/Generated/Models.cs"));

var document = await OpenApiYamlDocument.FromFileAsync(spec);
new EnumsAsStrings().Visit(document);

var settings = new CSharpClientGeneratorSettings
{
    GenerateClientClasses = false,
    GenerateClientInterfaces = false,
    GenerateExceptionClasses = false,
    GenerateDtoTypes = true,
};
var types = settings.CSharpGeneratorSettings;
types.Namespace = "GetStream.VisionAgents.Models";
types.JsonLibrary = CSharpJsonLibrary.SystemTextJson;
types.ClassStyle = CSharpClassStyle.Poco;
types.GenerateNullableReferenceTypes = true;
types.GenerateOptionalPropertiesAsNullable = true;
// A default copied into a request is how a caller silently loses what their config named.
types.GenerateDefaultValues = false;
types.GenerateDataAnnotations = false;
types.GenerateJsonMethods = false;
types.RequiredPropertiesMustBeDefined = false;
types.DateTimeType = "System.DateTimeOffset";
types.ArrayType = "System.Collections.Generic.List";
types.ArrayInstanceType = "System.Collections.Generic.List";
types.ArrayBaseType = "System.Collections.Generic.List";
types.DictionaryType = "System.Collections.Generic.Dictionary";
types.DictionaryInstanceType = "System.Collections.Generic.Dictionary";
types.DictionaryBaseType = "System.Collections.Generic.Dictionary";
types.PropertyNameGenerator = new PascalCaseNames();

var code = new CSharpClientGenerator(document, settings).GenerateFile().ReplaceLineEndings("\n");

if (args.Contains("--check"))
{
    var committed = File.Exists(output) ? File.ReadAllText(output) : "";
    if (committed != code)
    {
        Console.Error.WriteLine($"{output} is stale; run dotnet run --project tools/Generate");
        return 1;
    }
    return 0;
}

Directory.CreateDirectory(Path.GetDirectoryName(output)!);
File.WriteAllText(output, code);
Console.WriteLine($"wrote {output}");
return 0;

/// <summary>
/// Drops the value list from every enum, so it is generated as a string.
/// </summary>
/// <remarks>
/// A C# enum has nowhere to put a value it was not compiled with, so a state or a mode the
/// router adds after this ships would fail the whole response rather than one field of it.
/// The values stay in the description, where the doc comment shows them.
/// </remarks>
sealed class EnumsAsStrings : JsonSchemaVisitorBase
{
    protected override JsonSchema VisitSchema(JsonSchema schema, string path, string? typeNameHint)
    {
        if (schema.Enumeration.Count > 0 && schema.Type.HasFlag(JsonObjectType.String))
        {
            var values = string.Join(", ", schema.Enumeration.Select(value => $"`{value}`"));
            schema.Description = string.IsNullOrWhiteSpace(schema.Description)
                ? $"One of {values}."
                : $"{schema.Description.TrimEnd()}\n\nOne of {values}.";
            schema.Enumeration.Clear();
            schema.EnumerationNames.Clear();
        }
        return schema;
    }
}

sealed class PascalCaseNames : IPropertyNameGenerator
{
    public string Generate(JsonSchemaProperty property)
    {
        var words = property.Name.Split(['_', '-', '.', ' ', '$', '@'], StringSplitOptions.RemoveEmptyEntries);
        var name = string.Concat(words.Select(word => char.ToUpperInvariant(word[0]) + word[1..]));
        if (name.Length == 0 || char.IsDigit(name[0]))
        {
            name = "Value" + name;
        }
        return name;
    }
}
