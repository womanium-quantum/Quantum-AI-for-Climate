import React from 'react';
import { SearchBarCommands, SearchBarProps } from 'react-native-screens';
import { NativeProps as SearchBarNativeProps } from '../fabric/SearchBarNativeComponent';
export declare const NativeSearchBar: React.ComponentType<SearchBarNativeProps & {
    ref?: React.RefObject<SearchBarCommands>;
}> & typeof NativeSearchBarCommands;
export declare const NativeSearchBarCommands: SearchBarCommandsType;
type NativeSearchBarRef = React.ElementRef<typeof NativeSearchBar>;
type SearchBarCommandsType = {
    blur: (viewRef: NativeSearchBarRef) => void;
    focus: (viewRef: NativeSearchBarRef) => void;
    clearText: (viewRef: NativeSearchBarRef) => void;
    toggleCancelButton: (viewRef: NativeSearchBarRef, flag: boolean) => void;
    setText: (viewRef: NativeSearchBarRef, text: string) => void;
    cancelSearch: (viewRef: NativeSearchBarRef) => void;
};
declare const _default: React.ForwardRefExoticComponent<Omit<SearchBarProps, "ref"> & React.RefAttributes<SearchBarCommands>>;
export default _default;
//# sourceMappingURL=SearchBar.d.ts.map