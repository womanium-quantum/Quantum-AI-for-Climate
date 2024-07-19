import React from 'react';
import { View } from 'react-native';
import { ScreenProps } from 'react-native-screens';
export declare const NativeScreen: React.ComponentType<ScreenProps>;
export declare const InnerScreen: React.ForwardRefExoticComponent<Omit<ScreenProps, "ref"> & React.RefAttributes<View>>;
export declare const ScreenContext: React.Context<React.ForwardRefExoticComponent<Omit<ScreenProps, "ref"> & React.RefAttributes<View>>>;
declare const Screen: React.FC<ScreenProps>;
export default Screen;
//# sourceMappingURL=Screen.d.ts.map