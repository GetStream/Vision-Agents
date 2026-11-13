const path = require('path');
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

const sdkDir = process.env.STREAM_CHAT_RN_DIR;

if (sdkDir) {
  const sdkRoot = path.resolve(sdkDir);
  const sdkPackageDir = path.join(sdkRoot, 'package');
  const sdkPackages = {
    'stream-chat-expo': path.join(sdkPackageDir, 'expo-package'),
    'stream-chat-react-native-core': sdkPackageDir,
  };
  const appDependencies = new Set(Object.keys(require('./package.json').dependencies));
  const appOrigin = path.join(__dirname, 'package.json');
  const packageNameOf = (moduleName) =>
    moduleName.split('/').slice(0, moduleName.startsWith('@') ? 2 : 1).join('/');

  config.watchFolders = [sdkPackageDir, path.join(sdkRoot, 'node_modules')];
  config.resolver.resolveRequest = (context, moduleName, platform) => {
    const packageName = packageNameOf(moduleName);
    if (sdkPackages[packageName]) {
      const target = sdkPackages[packageName] + moduleName.slice(packageName.length);
      return context.resolveRequest(context, target, platform);
    }
    if (appDependencies.has(packageName) && context.originModulePath.startsWith(sdkRoot + path.sep)) {
      return context.resolveRequest({ ...context, originModulePath: appOrigin }, moduleName, platform);
    }
    return context.resolveRequest(context, moduleName, platform);
  };
}

module.exports = config;
