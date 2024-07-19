function _extends() { _extends = Object.assign ? Object.assign.bind() : function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return _extends.apply(this, arguments); }
import React from 'react';
import { isSearchBarAvailableForCurrentPlatform } from 'react-native-screens';
import { View } from 'react-native';

// Native components
import SearchBarNativeComponent, { Commands as SearchBarNativeCommands } from '../fabric/SearchBarNativeComponent';
export const NativeSearchBar = SearchBarNativeComponent;
export const NativeSearchBarCommands = SearchBarNativeCommands;
function SearchBar(props, ref) {
  const searchBarRef = React.useRef(null);
  React.useImperativeHandle(ref, () => ({
    blur: () => {
      _callMethodWithRef(ref => NativeSearchBarCommands.blur(ref));
    },
    focus: () => {
      _callMethodWithRef(ref => NativeSearchBarCommands.focus(ref));
    },
    toggleCancelButton: flag => {
      _callMethodWithRef(ref => NativeSearchBarCommands.toggleCancelButton(ref, flag));
    },
    clearText: () => {
      _callMethodWithRef(ref => NativeSearchBarCommands.clearText(ref));
    },
    setText: text => {
      _callMethodWithRef(ref => NativeSearchBarCommands.setText(ref, text));
    },
    cancelSearch: () => {
      _callMethodWithRef(ref => NativeSearchBarCommands.cancelSearch(ref));
    }
  }));
  const _callMethodWithRef = React.useCallback(method => {
    const ref = searchBarRef.current;
    if (ref) {
      method(ref);
    } else {
      console.warn('Reference to native search bar component has not been updated yet');
    }
  }, [searchBarRef]);
  if (!isSearchBarAvailableForCurrentPlatform) {
    console.warn('Importing SearchBar is only valid on iOS and Android devices.');
    return View;
  }
  return /*#__PURE__*/React.createElement(NativeSearchBar, _extends({
    ref: searchBarRef
  }, props, {
    onSearchFocus: props.onFocus,
    onSearchBlur: props.onBlur,
    onSearchButtonPress: props.onSearchButtonPress,
    onCancelButtonPress: props.onCancelButtonPress,
    onChangeText: props.onChangeText
  }));
}
export default /*#__PURE__*/React.forwardRef(SearchBar);
//# sourceMappingURL=SearchBar.js.map