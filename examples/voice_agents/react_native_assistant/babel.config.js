module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: [
      // Needed by `stream-chat-expo/ai-components`: its `zod` dependency ships `export * as`.
      '@babel/plugin-transform-export-namespace-from',
      // react-native-worklets/plugin has to be the last one
      'react-native-worklets/plugin',
    ],
  };
};
