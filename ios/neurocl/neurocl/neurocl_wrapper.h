//
//  neurocl_wrapper.h
//  neurocl
//
//  Created by Albert Murienne on 14/03/2017.
//  Copyright © 2017 Blackccpie. All rights reserved.
//

#ifndef neurocl_wrapper_h
#define neurocl_wrapper_h

#import <UIKit/UIKit.h>

@interface NeuroclWrapper : NSObject

- (instancetype) init;
- (instancetype) initWithNet:(NSString*) topology weights:(NSString*) weights;
- (void) dealloc;
- (NSString*) digit_recognizer:(UIImage*) in;

@end

#endif /* neurocl_wrapper_h */
